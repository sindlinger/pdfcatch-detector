from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from pdfcatch.qwen_supervisionado._internal.paths import QWEN_INPUT_ROOT, bootstrap_layout, ensure_inside


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_label(s: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in str(s).strip().lower())
    out = "_".join(p for p in out.split("_") if p)
    if not out:
        raise ValueError("label invalida")
    return out


def _list_pdfs_recursive(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        raise ValueError(f"diretorio invalido: {root}")
    return sorted(p.resolve() for p in root.rglob("*.pdf") if p.is_file())


def _split_counts(n: int, train_pct: float, val_pct: float, test_pct: float) -> tuple[int, int, int]:
    if n <= 0:
        return (0, 0, 0)
    train = int(math.floor(n * train_pct))
    val = int(math.floor(n * val_pct))
    used = train + val
    test = n - used
    return (train, val, test)


def _iter_labeled_rows(files: list[Path], label: str, split_name: str) -> Iterable[dict[str, str]]:
    for i, src in enumerate(files, start=1):
        digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:10]
        dst_name = f"{i:05d}_{digest}_{src.name}"
        yield {
            "label": label,
            "split": split_name,
            "src_abs": str(src),
            "src_name": src.name,
            "dst_name": dst_name,
        }


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def run(
    *,
    positive_dir: Path,
    negative_dir: Path,
    out_dataset_dir: Path,
    positive_label: str,
    negative_label: str,
    seed: int,
    balance: bool,
    train_pct: float,
    val_pct: float,
    test_pct: float,
    overwrite: bool,
) -> int:
    bootstrap_layout()
    out_root = ensure_inside(out_dataset_dir, QWEN_INPUT_ROOT, "dataset supervisionado")
    pos_label = _normalize_label(positive_label)
    neg_label = _normalize_label(negative_label)

    pos_all = _list_pdfs_recursive(positive_dir)
    neg_all = _list_pdfs_recursive(negative_dir)
    if not pos_all:
        raise ValueError(f"nenhum positivo encontrado em: {positive_dir}")
    if not neg_all:
        raise ValueError(f"nenhum negativo encontrado em: {negative_dir}")

    rng = random.Random(int(seed))
    pos = list(pos_all)
    neg = list(neg_all)
    rng.shuffle(pos)
    rng.shuffle(neg)

    if balance:
        n = min(len(pos), len(neg))
        pos = pos[:n]
        neg = neg[:n]

    if out_root.exists() and any(out_root.iterdir()) and not overwrite:
        raise ValueError(
            f"dataset ja existe e nao esta vazio: {out_root}. "
            "use --overwrite para recriar."
        )

    if overwrite and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    pos_train_n, pos_val_n, pos_test_n = _split_counts(len(pos), train_pct, val_pct, test_pct)
    neg_train_n, neg_val_n, neg_test_n = _split_counts(len(neg), train_pct, val_pct, test_pct)

    pos_train = pos[:pos_train_n]
    pos_val = pos[pos_train_n : pos_train_n + pos_val_n]
    pos_test = pos[pos_train_n + pos_val_n :]

    neg_train = neg[:neg_train_n]
    neg_val = neg[neg_train_n : neg_train_n + neg_val_n]
    neg_test = neg[neg_train_n + neg_val_n :]

    manifest_rows: list[dict[str, str]] = []
    plan = [
        (pos_train, pos_label, "train"),
        (pos_val, pos_label, "val"),
        (pos_test, pos_label, "test"),
        (neg_train, neg_label, "train"),
        (neg_val, neg_label, "val"),
        (neg_test, neg_label, "test"),
    ]

    for files, label, split in plan:
        for row in _iter_labeled_rows(files, label, split):
            dst = out_root / split / label / row["dst_name"]
            _copy_file(Path(row["src_abs"]), dst)
            row["dst_abs"] = str(dst.resolve())
            manifest_rows.append(row)

    manifest_csv = out_root / "manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["label", "split", "src_abs", "src_name", "dst_name", "dst_abs"],
        )
        w.writeheader()
        for r in manifest_rows:
            w.writerow(r)

    summary = {
        "status": "ok",
        "scope": "qwen_isolated",
        "operation": "prepare_supervised_dataset",
        "created_at_utc": _now_utc_iso(),
        "dataset_root": str(out_root.resolve()),
        "source_positive_dir": str(positive_dir.resolve()),
        "source_negative_dir": str(negative_dir.resolve()),
        "labels": {"positive": pos_label, "negative": neg_label},
        "balance": bool(balance),
        "seed": int(seed),
        "split_pct": {
            "train": train_pct,
            "val": val_pct,
            "test": test_pct,
        },
        "counts_source": {
            "positive_total": len(pos_all),
            "negative_total": len(neg_all),
        },
        "counts_selected": {
            "positive": len(pos),
            "negative": len(neg),
        },
        "counts_split": {
            "train": {pos_label: len(pos_train), neg_label: len(neg_train)},
            "val": {pos_label: len(pos_val), neg_label: len(neg_val)},
            "test": {pos_label: len(pos_test), neg_label: len(neg_test)},
        },
        "manifest_csv": str(manifest_csv.resolve()),
    }

    summary_json = out_root / "summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Prepara dataset supervisionado (positivo/negativo + train/val/test) "
            "para treino real, gravando somente em io/qwen_isolated/input."
        )
    )
    ap.add_argument("--positive-dir", required=True, help="Diretorio origem de positivos")
    ap.add_argument("--negative-dir", required=True, help="Diretorio origem de negativos")
    ap.add_argument(
        "--out-dataset-dir",
        required=True,
        help="Destino dentro de io/qwen_isolated/input",
    )
    ap.add_argument("--positive-label", default="despacho")
    ap.add_argument("--negative-label", default="nao_despacho")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balance", action="store_true", help="Balanceia classes por menor cardinalidade")
    ap.add_argument("--train-pct", type=float, default=0.8)
    ap.add_argument("--val-pct", type=float, default=0.1)
    ap.add_argument("--test-pct", type=float, default=0.1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    train_pct = float(args.train_pct)
    val_pct = float(args.val_pct)
    test_pct = float(args.test_pct)
    if train_pct <= 0 or val_pct < 0 or test_pct < 0:
        raise ValueError("percentuais invalidos")
    s = train_pct + val_pct + test_pct
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"percentuais devem somar 1.0 (recebido {s})")

    return run(
        positive_dir=Path(args.positive_dir),
        negative_dir=Path(args.negative_dir),
        out_dataset_dir=Path(args.out_dataset_dir),
        positive_label=str(args.positive_label),
        negative_label=str(args.negative_label),
        seed=int(args.seed),
        balance=bool(args.balance),
        train_pct=train_pct,
        val_pct=val_pct,
        test_pct=test_pct,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    raise SystemExit(main())

