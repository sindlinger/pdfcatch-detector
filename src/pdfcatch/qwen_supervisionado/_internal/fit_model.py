from __future__ import annotations

import argparse
import copy
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import numpy as np

from pdfcatch.catalogacao_classica._internal.features import extract_page_text_from_doc
from pdfcatch.qwen_supervisionado._internal.paths import (
    QWEN_INPUT_ROOT,
    QWEN_MODEL_ROOT,
    QWEN_OUTPUT_ROOT,
    bootstrap_layout,
    ensure_inside,
    ensure_model_dir,
    ensure_output_file,
)

CATALOG_PAGE_INDEX = 0


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _load_sentence_transformer_gpu(model_name: str):
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise ValueError(
            "treino supervisionado exige GPU: pacote 'torch' nao encontrado."
        ) from exc
    if not bool(torch.cuda.is_available()):
        raise ValueError(
            "treino supervisionado oficial exige GPU CUDA ativa. "
            "fallback para CPU nao e permitido."
        )
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        raise ValueError(
            "treino supervisionado exige pacote 'sentence-transformers'."
        ) from exc
    try:
        try:
            model = SentenceTransformer(str(model_name), device="cuda", trust_remote_code=True)
        except TypeError:
            model = SentenceTransformer(str(model_name), device="cuda")
    except Exception as exc:
        raise ValueError(
            f"falha ao carregar modelo de embeddings em GPU: model={model_name} erro={exc}"
        ) from exc
    return model


def _read_page_text(pdf: Path, page_index: int) -> str:
    doc = fitz.open(pdf)
    try:
        if page_index < 0 or page_index >= int(doc.page_count):
            return ""
        txt = extract_page_text_from_doc(doc, page_index)
        return _normalize_text(txt)
    finally:
        try:
            doc.close()
        except Exception:
            pass


@dataclass
class Sample:
    pdf: Path
    label: str
    text: str


def _collect_split(dataset_root: Path, split: str, min_chars: int) -> tuple[list[Sample], list[dict[str, str]]]:
    split_dir = dataset_root / split
    if not split_dir.exists() or not split_dir.is_dir():
        return ([], [])
    labels = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    out: list[Sample] = []
    skipped: list[dict[str, str]] = []
    for label in labels:
        for pdf in sorted((split_dir / label).rglob("*.pdf")):
            txt = _read_page_text(pdf, CATALOG_PAGE_INDEX)
            if len(txt) < int(min_chars):
                skipped.append(
                    {
                        "split": split,
                        "label": label,
                        "pdf": str(pdf.resolve()),
                        "reason": f"text_too_short({len(txt)}<{int(min_chars)})",
                    }
                )
                continue
            out.append(Sample(pdf=pdf.resolve(), label=label, text=txt))
    return (out, skipped)


def _encode_texts(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    cur_bs = max(1, int(batch_size))
    while True:
        try:
            vecs = model.encode(
                texts,
                batch_size=cur_bs,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(vecs, dtype=np.float32)
        except Exception as exc:
            msg = str(exc).lower()
            if ("out of memory" in msg or "cuda oom" in msg) and cur_bs > 1:
                cur_bs = max(1, cur_bs // 2)
                print(json.dumps({"event": "oom_retry", "next_batch_size": cur_bs}, ensure_ascii=False))
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            raise


def _to_xy(samples: list[Sample], label_to_id: dict[str, int], emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(samples) != int(emb.shape[0]):
        raise ValueError("tamanho de embeddings incompativel com amostras")
    y = np.array([int(label_to_id[s.label]) for s in samples], dtype=np.int64)
    return (emb, y)


def _metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, float]:
    total = int(len(y_true))
    if total == 0:
        return {"accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}
    acc = float(np.mean((y_true == y_pred).astype(np.float32)))
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for c in range(int(num_classes)):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        p = (float(tp) / float(tp + fp)) if (tp + fp) > 0 else 0.0
        r = (float(tp) / float(tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    return {
        "accuracy": round(acc, 6),
        "macro_precision": round(float(np.mean(precisions)), 6),
        "macro_recall": round(float(np.mean(recalls)), 6),
        "macro_f1": round(float(np.mean(f1s)), 6),
    }


def _evaluate_torch(model: Any, x: np.ndarray, y: np.ndarray, batch_size: int, device: str) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if x.size == 0 or y.size == 0:
        return {
            "loss": None,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "count": 0,
        }

    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    dl = DataLoader(ds, batch_size=max(1, int(batch_size)), shuffle=False)
    ce = nn.CrossEntropyLoss()
    model.eval()
    losses: list[float] = []
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb)
            losses.append(float(loss.detach().cpu().item()))
            y_hat = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
            preds.append(y_hat)
            trues.append(yb.detach().cpu().numpy().astype(np.int64))
    y_pred = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.int64)
    y_true = np.concatenate(trues, axis=0) if trues else np.zeros((0,), dtype=np.int64)
    m = _metrics_from_predictions(y_true=y_true, y_pred=y_pred, num_classes=int(model.num_classes))
    m["loss"] = round(float(np.mean(losses)), 6) if losses else None
    m["count"] = int(y_true.size)
    return m


class _LinearClassifier:
    def __init__(self, embedding_dim: int, num_classes: int, dropout: float):
        import torch
        from torch import nn

        class _Net(nn.Module):
            def __init__(self, d: int, k: int, p: float):
                super().__init__()
                self.dropout = nn.Dropout(float(p))
                self.fc = nn.Linear(int(d), int(k))
                self.num_classes = int(k)

            def forward(self, x):
                return self.fc(self.dropout(x))

        self.model = _Net(int(embedding_dim), int(num_classes), float(dropout))
        self._torch = torch

    def to(self, device: str):
        self.model.to(device)
        return self

    @property
    def num_classes(self) -> int:
        return int(self.model.num_classes)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.model.load_state_dict(sd)


def run(
    *,
    dataset_dir: Path,
    model_out_dir: Path,
    report_out: Path,
    embedding_model: str,
    min_chars: int,
    batch_size_embed: int,
    batch_size_train: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    seed: int,
    max_text_chars: int,
    max_seq_length: int,
) -> int:
    bootstrap_layout()
    _set_seed(int(seed))

    ds_root = ensure_inside(dataset_dir, QWEN_INPUT_ROOT, "dataset supervisionado")
    if not ds_root.exists() or not ds_root.is_dir():
        raise ValueError(f"dataset supervisionado invalido: {ds_root}")

    out_model_dir = ensure_model_dir(model_out_dir)
    out_report = ensure_output_file(report_out)

    train_samples, skipped_train = _collect_split(ds_root, "train", int(min_chars))
    val_samples, skipped_val = _collect_split(ds_root, "val", int(min_chars))
    test_samples, skipped_test = _collect_split(ds_root, "test", int(min_chars))
    if not train_samples:
        raise ValueError("split train vazio apos filtros de texto")

    labels = sorted({s.label for s in train_samples})
    if len(labels) < 2:
        raise ValueError("treino supervisionado exige pelo menos 2 classes no split train")
    label_to_id = {lab: i for i, lab in enumerate(labels)}

    embedder = _load_sentence_transformer_gpu(str(embedding_model))
    try:
        embedder.max_seq_length = int(max_seq_length)
    except Exception:
        pass
    train_texts = [s.text[: int(max_text_chars)] for s in train_samples]
    val_texts = [s.text[: int(max_text_chars)] for s in val_samples]
    test_texts = [s.text[: int(max_text_chars)] for s in test_samples]
    emb_train = _encode_texts(embedder, train_texts, int(batch_size_embed))
    emb_val = _encode_texts(embedder, val_texts, int(batch_size_embed)) if val_samples else np.zeros((0, emb_train.shape[1]), dtype=np.float32)
    emb_test = _encode_texts(embedder, test_texts, int(batch_size_embed)) if test_samples else np.zeros((0, emb_train.shape[1]), dtype=np.float32)

    x_train, y_train = _to_xy(train_samples, label_to_id, emb_train)
    x_val, y_val = _to_xy(val_samples, label_to_id, emb_val) if val_samples else (np.zeros((0, emb_train.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64))
    x_test, y_test = _to_xy(test_samples, label_to_id, emb_test) if test_samples else (np.zeros((0, emb_train.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64))

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda"
    clf = _LinearClassifier(embedding_dim=int(x_train.shape[1]), num_classes=len(labels), dropout=float(dropout))
    clf.to(device)

    ds_train = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    dl_train = DataLoader(ds_train, batch_size=max(1, int(batch_size_train)), shuffle=True)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(clf.model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))

    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_key = -1.0
    epoch_rows: list[dict[str, Any]] = []

    for ep in range(1, int(epochs) + 1):
        clf.model.train()
        losses: list[float] = []
        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = clf.model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        train_m = _evaluate_torch(clf.model, x_train, y_train, int(batch_size_train), device)
        val_m = _evaluate_torch(clf.model, x_val, y_val, int(batch_size_train), device)
        key = float(val_m.get("macro_f1") if val_m.get("count", 0) > 0 else train_m.get("macro_f1") or 0.0)
        if key >= best_key:
            best_key = key
            best_epoch = int(ep)
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in clf.state_dict().items()})
        row = {
            "epoch": int(ep),
            "train_loss_batch_mean": round(float(np.mean(losses)), 6) if losses else None,
            "train": train_m,
            "val": val_m,
            "selection_key_macro_f1": round(float(key), 6),
        }
        epoch_rows.append(row)
        print(
            json.dumps(
                {
                    "epoch": int(ep),
                    "train_macro_f1": train_m.get("macro_f1"),
                    "val_macro_f1": val_m.get("macro_f1"),
                    "train_acc": train_m.get("accuracy"),
                    "val_acc": val_m.get("accuracy"),
                },
                ensure_ascii=False,
            )
        )

    if best_state is None:
        raise RuntimeError("falha interna: estado final do modelo nao foi capturado")
    clf.load_state_dict(best_state)
    clf.to(device)

    metrics_train = _evaluate_torch(clf.model, x_train, y_train, int(batch_size_train), device)
    metrics_val = _evaluate_torch(clf.model, x_val, y_val, int(batch_size_train), device)
    metrics_test = _evaluate_torch(clf.model, x_test, y_test, int(batch_size_train), device)

    model_pt = out_model_dir / "classifier.pt"
    meta_json = out_model_dir / "model_meta.json"

    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in clf.state_dict().items()},
            "embedding_dim": int(x_train.shape[1]),
            "labels": labels,
            "label_to_id": label_to_id,
            "page_index": CATALOG_PAGE_INDEX,
        },
        model_pt,
    )

    meta = {
        "schema": "qwen_supervised_classifier.v1",
        "created_at_utc": _now_utc_iso(),
        "device_required": "cuda",
        "page_index": CATALOG_PAGE_INDEX,
        "dataset_dir": str(ds_root.resolve()),
        "embedding_model": str(embedding_model),
        "artifacts": {
            "model_dir": str(out_model_dir.resolve()),
            "classifier_pt": str(model_pt.resolve()),
            "model_meta_json": str(meta_json.resolve()),
            "report_json": str(out_report.resolve()),
        },
        "labels": labels,
        "label_to_id": label_to_id,
        "min_chars": int(min_chars),
        "max_text_chars": int(max_text_chars),
        "max_seq_length": int(max_seq_length),
        "embedding_dim": int(x_train.shape[1]),
        "train_config": {
            "batch_size_embed": int(batch_size_embed),
            "batch_size_train": int(batch_size_train),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "dropout": float(dropout),
            "seed": int(seed),
        },
        "counts": {
            "train": int(len(train_samples)),
            "val": int(len(val_samples)),
            "test": int(len(test_samples)),
            "skipped_train": int(len(skipped_train)),
            "skipped_val": int(len(skipped_val)),
            "skipped_test": int(len(skipped_test)),
        },
        "best_epoch": int(best_epoch),
        "metrics": {
            "train": metrics_train,
            "val": metrics_val,
            "test": metrics_test,
        },
        "epochs": epoch_rows,
        "skipped_examples_preview": {
            "train": skipped_train[:50],
            "val": skipped_val[:50],
            "test": skipped_test[:50],
        },
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_report.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "operation": "fit_model",
                "scope": "qwen_isolated",
                "model_dir": str(out_model_dir.resolve()),
                "report_json": str(out_report.resolve()),
                "best_epoch": int(best_epoch),
                "test_macro_f1": metrics_test.get("macro_f1"),
                "test_accuracy": metrics_test.get("accuracy"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Treino supervisionado em GPU (Qwen): treina classificador sobre embeddings "
            "do dataset positivo/negativo preparado no io/qwen_isolated/input."
        )
    )
    ap.add_argument("--dataset-dir", required=True, help="Dataset dentro de io/qwen_isolated/input")
    ap.add_argument("--embedding-model", required=True, help="Modelo de embedding (ex: Qwen/Qwen3-Embedding-0.6B)")
    ap.add_argument("--model-out-dir", default=None, help="Diretorio de saida dentro de io/qwen_isolated/models")
    ap.add_argument("--report-out", default=None, help="JSON de relatorio em io/qwen_isolated/outputs")
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--batch-size-embed", type=int, default=64)
    ap.add_argument("--batch-size-train", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-text-chars", type=int, default=3000)
    ap.add_argument("--max-seq-length", type=int, default=384)
    args = ap.parse_args(argv)

    ts = _now_compact()
    ds = Path(args.dataset_dir)
    out_model_dir = (
        Path(args.model_out_dir)
        if args.model_out_dir
        else QWEN_MODEL_ROOT / f"supervised_{ds.name}_{ts}"
    )
    out_report = (
        Path(args.report_out)
        if args.report_out
        else QWEN_OUTPUT_ROOT / f"fit_model_{ds.name}_{ts}.json"
    )
    return run(
        dataset_dir=ds,
        model_out_dir=out_model_dir,
        report_out=out_report,
        embedding_model=str(args.embedding_model),
        min_chars=int(args.min_chars),
        batch_size_embed=int(args.batch_size_embed),
        batch_size_train=int(args.batch_size_train),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        dropout=float(args.dropout),
        seed=int(args.seed),
        max_text_chars=int(args.max_text_chars),
        max_seq_length=int(args.max_seq_length),
    )


if __name__ == "__main__":
    raise SystemExit(main())
