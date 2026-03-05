from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
import numpy as np

from pdfcatch.catalogacao_classica._internal.compare_with_paradigm import _load_sentence_transformer_gpu, _load_paradigm
from pdfcatch.catalogacao_classica._internal.features import extract_page_text_from_doc
from pdfcatch.qwen_supervisionado._internal.catalog_store import CATALOG_INDEX_PATH, get_tag_entry, normalize_tag
from pdfcatch.qwen_supervisionado._internal.paths import (
    QWEN_CATALOG_ROOT,
    QWEN_OUTPUT_ROOT,
    bootstrap_layout,
    ensure_inside,
    ensure_input_dir,
    ensure_output_file,
)

CATALOG_PAGE_INDEX = 0


def _now_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _read_page_text(pdf: Path, page_index: int) -> str:
    doc = fitz.open(pdf)
    try:
        if page_index < 0 or page_index >= int(doc.page_count):
            return ""
        return _normalize_text(extract_page_text_from_doc(doc, page_index))
    finally:
        try:
            doc.close()
        except Exception:
            pass


def run(
    *,
    tag: str,
    dir_path: Path,
    out_path: Path,
    min_chars: int,
    threshold_override_pct: float | None,
) -> int:
    bootstrap_layout()
    source_dir = ensure_input_dir(dir_path)

    norm_tag = normalize_tag(tag)
    entry = get_tag_entry(norm_tag, CATALOG_INDEX_PATH)
    if entry is None:
        raise ValueError(f"tag nao catalogada: {norm_tag}")

    paradigm_raw = str(entry.get("paradigm_npz") or "")
    paradigm_npz = ensure_inside(paradigm_raw, QWEN_CATALOG_ROOT, "paradigma catalogado")
    if not paradigm_npz.exists():
        raise ValueError(f"paradigma da tag nao encontrado: {paradigm_npz}")

    model_name = str(entry.get("model_name") or "").strip()
    if not model_name:
        raise ValueError(f"model_name ausente no catalogo da tag: {norm_tag}")

    paradigm, threshold_raw, _src = _load_paradigm(paradigm_npz)
    threshold = threshold_raw
    if threshold_override_pct is not None:
        threshold = float(threshold_override_pct) / 100.0

    pdfs = sorted([p for p in source_dir.glob("*.pdf") if p.is_file()])
    if not pdfs:
        raise ValueError(f"nenhum PDF encontrado em: {source_dir}")

    rows: list[dict[str, Any]] = []
    texts: list[str] = []
    valid_idx: list[int] = []

    for i, pdf in enumerate(pdfs):
        txt = _read_page_text(pdf, CATALOG_PAGE_INDEX)
        chars = len(txt)
        row = {
            "file": pdf.name,
            "pdf": str(pdf.resolve()),
            "page_index": CATALOG_PAGE_INDEX,
            "chars": int(chars),
            "status": "ok" if chars >= int(min_chars) else "skipped_short_text",
            "similarity_pct": None,
            "pass_threshold": None,
        }
        rows.append(row)
        if chars >= int(min_chars):
            valid_idx.append(i)
            texts.append(txt)

    if texts:
        model = _load_sentence_transformer_gpu(model_name)
        vecs = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vecs = np.asarray(vecs, dtype=np.float32)

        for j, row_i in enumerate(valid_idx):
            sim = float(np.dot(vecs[j], paradigm))
            rows[row_i]["similarity_pct"] = round(sim * 100.0, 4)
            if threshold is not None:
                rows[row_i]["pass_threshold"] = bool(sim >= float(threshold))
            else:
                rows[row_i]["pass_threshold"] = None

    found = [r for r in rows if r.get("pass_threshold") is True]
    found_sorted = sorted(found, key=lambda r: float(r.get("similarity_pct") or 0.0), reverse=True)

    output_file = ensure_output_file(out_path)
    payload = {
        "status": "ok",
        "scope": "qwen_isolated",
        "operation": "find_docs",
        "tag": norm_tag,
        "model_name": model_name,
        "catalog_index": str(CATALOG_INDEX_PATH.resolve()),
        "paradigm_npz": str(paradigm_npz),
        "dir": str(source_dir),
        "page_index": CATALOG_PAGE_INDEX,
        "threshold_pct": (round(float(threshold) * 100.0, 4) if threshold is not None else None),
        "total_files": len(rows),
        "found_count": len(found_sorted),
        "found": found_sorted,
        "all_results": rows,
    }

    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "scope": "qwen_isolated",
        "operation": "find_docs",
        "tag": norm_tag,
        "out": str(output_file.resolve()),
        "total_files": len(rows),
        "found_count": len(found_sorted),
        "threshold_pct": payload["threshold_pct"],
    }, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "find_docs (Qwen isolado): inferencia por TAG catalogada, "
            "comparando sempre a pagina 0 e gravando apenas em io/qwen_isolated/outputs."
        )
    )
    ap.add_argument("--tag", required=True, help="TAG catalogada (ex: despacho)")
    ap.add_argument(
        "--dir",
        required=True,
        help="Diretorio de entrada dentro de io/qwen_isolated/input",
    )
    ap.add_argument("--min-chars", type=int, default=40, help="Minimo de caracteres para comparar")
    ap.add_argument(
        "--threshold-override-pct",
        type=float,
        default=None,
        help="Override do limiar em percentual (ex: 90.0)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="JSON de saida dentro de io/qwen_isolated/outputs",
    )
    args = ap.parse_args(argv)

    norm_tag = normalize_tag(str(args.tag))
    out = (
        Path(args.out)
        if args.out
        else QWEN_OUTPUT_ROOT / f"find_docs_{norm_tag}_{_now_compact()}.json"
    )
    return run(
        tag=norm_tag,
        dir_path=Path(args.dir),
        out_path=out,
        min_chars=int(args.min_chars),
        threshold_override_pct=(float(args.threshold_override_pct) if args.threshold_override_pct is not None else None),
    )


if __name__ == "__main__":
    raise SystemExit(main())

