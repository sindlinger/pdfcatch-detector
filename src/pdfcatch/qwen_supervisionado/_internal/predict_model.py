from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
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
    ensure_input_dir,
    ensure_inside,
    ensure_output_file,
)


def _now_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _load_sentence_transformer_gpu(model_name: str):
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise ValueError(
            "inferencia supervisionada exige GPU: pacote 'torch' nao encontrado."
        ) from exc
    if not bool(torch.cuda.is_available()):
        raise ValueError(
            "inferencia supervisionada oficial exige GPU CUDA ativa. "
            "fallback para CPU nao e permitido."
        )
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        raise ValueError(
            "inferencia supervisionada exige pacote 'sentence-transformers'."
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


def _load_model(model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    d = ensure_inside(model_dir, QWEN_MODEL_ROOT, "diretorio de modelo")
    if not d.exists() or not d.is_dir():
        raise ValueError(f"diretorio de modelo invalido: {d}")
    meta_path = ensure_inside(d / "model_meta.json", QWEN_MODEL_ROOT, "meta de modelo")
    pt_path = ensure_inside(d / "classifier.pt", QWEN_MODEL_ROOT, "checkpoint de modelo")
    if not meta_path.exists() or not meta_path.is_file():
        raise ValueError(f"model_meta.json ausente em: {d}")
    if not pt_path.exists() or not pt_path.is_file():
        raise ValueError(f"classifier.pt ausente em: {d}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ckpt = torch.load(pt_path, map_location="cpu")
    return (meta, ckpt)


def _build_classifier(embedding_dim: int, num_classes: int, dropout: float):
    import torch
    from torch import nn

    class _Net(nn.Module):
        def __init__(self, d: int, k: int, p: float):
            super().__init__()
            self.dropout = nn.Dropout(float(p))
            self.fc = nn.Linear(int(d), int(k))

        def forward(self, x):
            return self.fc(self.dropout(x))

    net = _Net(embedding_dim, num_classes, dropout)
    net.to("cuda")
    net.eval()
    return net


def run(
    *,
    model_dir: Path,
    pdf: Path | None,
    dir_path: Path | None,
    out_path: Path,
    batch_size_embed: int,
) -> int:
    bootstrap_layout()
    meta, ckpt = _load_model(model_dir)
    labels = list(meta.get("labels") or [])
    if not labels:
        raise ValueError("labels ausentes no model_meta.json")
    label_to_id = dict(meta.get("label_to_id") or {})
    if not label_to_id:
        raise ValueError("label_to_id ausente no model_meta.json")
    embedding_model = str(meta.get("embedding_model") or "").strip()
    if not embedding_model:
        raise ValueError("embedding_model ausente no model_meta.json")
    min_chars = int(meta.get("min_chars") or 20)
    max_text_chars = int(meta.get("max_text_chars") or 3000)
    max_seq_length = int(meta.get("max_seq_length") or 384)
    page_index = int(meta.get("page_index") or 0)
    train_cfg = dict(meta.get("train_config") or {})
    dropout = float(train_cfg.get("dropout") or 0.1)
    embedding_dim = int(meta.get("embedding_dim") or 0)
    if embedding_dim <= 0:
        raise ValueError("embedding_dim invalido no model_meta.json")

    pdfs: list[Path]
    if pdf is not None:
        p = ensure_inside(pdf, QWEN_INPUT_ROOT, "pdf de entrada")
        if not p.exists() or not p.is_file():
            raise ValueError(f"pdf invalido: {p}")
        pdfs = [p]
    else:
        if dir_path is None:
            raise ValueError("informe --pdf ou --dir")
        d = ensure_input_dir(dir_path)
        pdfs = sorted([p.resolve() for p in d.glob("*.pdf") if p.is_file()])
        if not pdfs:
            raise ValueError(f"nenhum PDF encontrado em: {d}")

    rows: list[dict[str, Any]] = []
    valid_texts: list[str] = []
    valid_indices: list[int] = []
    for i, p in enumerate(pdfs):
        txt = _read_page_text(p, page_index)
        chars = len(txt)
        row = {
            "file": p.name,
            "pdf": str(p.resolve()),
            "page_index": page_index,
            "chars": chars,
            "status": "ok" if chars >= int(min_chars) else "skipped_short_text",
            "predicted_label": None,
            "confidence_pct": None,
            "probs_pct": {},
        }
        rows.append(row)
        if chars >= int(min_chars):
            valid_indices.append(i)
            valid_texts.append(txt[: int(max_text_chars)])

    if valid_texts:
        embedder = _load_sentence_transformer_gpu(embedding_model)
        try:
            embedder.max_seq_length = int(max_seq_length)
        except Exception:
            pass
        vecs = embedder.encode(
            valid_texts,
            batch_size=max(1, int(batch_size_embed)),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        emb = np.asarray(vecs, dtype=np.float32)
        if int(emb.shape[1]) != int(embedding_dim):
            raise ValueError(
                f"dimensao de embedding divergente: checkpoint={embedding_dim} atual={int(emb.shape[1])}"
            )

        import torch

        clf = _build_classifier(embedding_dim=embedding_dim, num_classes=len(labels), dropout=dropout)
        sd = dict(ckpt.get("state_dict") or {})
        if not sd:
            raise ValueError("state_dict ausente no checkpoint")
        clf.load_state_dict(sd)
        with torch.no_grad():
            xb = torch.from_numpy(emb).float().to("cuda")
            logits = clf(xb)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
        for j, row_i in enumerate(valid_indices):
            pr = probs[j]
            best_id = int(np.argmax(pr))
            best_label = str(labels[best_id])
            rows[row_i]["predicted_label"] = best_label
            rows[row_i]["confidence_pct"] = round(float(pr[best_id]) * 100.0, 4)
            rows[row_i]["probs_pct"] = {
                str(labels[k]): round(float(pr[k]) * 100.0, 4) for k in range(int(pr.shape[0]))
            }

    out_file = ensure_output_file(out_path)
    payload = {
        "status": "ok",
        "scope": "qwen_isolated",
        "operation": "predict_model",
        "model_dir": str(ensure_inside(model_dir, QWEN_MODEL_ROOT, "diretorio de modelo")),
        "embedding_model": embedding_model,
        "page_index": page_index,
        "min_chars": min_chars,
        "max_text_chars": max_text_chars,
        "max_seq_length": max_seq_length,
        "labels": labels,
        "total_files": len(rows),
        "predicted_count": int(sum(1 for r in rows if r.get("predicted_label"))),
        "rows": rows,
    }
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "scope": "qwen_isolated",
                "operation": "predict_model",
                "out": str(out_file.resolve()),
                "total_files": len(rows),
                "predicted_count": payload["predicted_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Inferencia supervisionada em GPU para checkpoint treinado em fit_model. "
            "Compara somente pagina 0."
        )
    )
    ap.add_argument("--model-dir", required=True, help="Diretorio do checkpoint em io/qwen_isolated/models")
    ap.add_argument("--pdf", default=None, help="PDF unico dentro de io/qwen_isolated/input")
    ap.add_argument("--dir", default=None, help="Diretorio de PDFs dentro de io/qwen_isolated/input")
    ap.add_argument("--out", default=None, help="JSON em io/qwen_isolated/outputs")
    ap.add_argument("--batch-size-embed", type=int, default=128)
    args = ap.parse_args(argv)

    if bool(args.pdf) == bool(args.dir):
        raise ValueError("informe exatamente um: --pdf ou --dir")

    ts = _now_compact()
    out = Path(args.out) if args.out else QWEN_OUTPUT_ROOT / f"predict_model_{ts}.json"
    return run(
        model_dir=Path(args.model_dir),
        pdf=(Path(args.pdf) if args.pdf else None),
        dir_path=(Path(args.dir) if args.dir else None),
        out_path=out,
        batch_size_embed=int(args.batch_size_embed),
    )


if __name__ == "__main__":
    raise SystemExit(main())
