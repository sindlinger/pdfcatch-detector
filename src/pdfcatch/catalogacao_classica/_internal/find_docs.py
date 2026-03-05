from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
import numpy as np

from pdfcatch.catalogacao_classica._internal.catalog_store import CATALOG_INDEX_PATH, get_tag_entry, normalize_tag
from pdfcatch.catalogacao_classica._internal.compare_with_paradigm import _load_sentence_transformer_gpu, _load_paradigm
from pdfcatch.catalogacao_classica._internal.features import extract_page_text_from_doc


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
    page_index: int,
    min_chars: int,
    threshold_override_pct: float | None,
) -> int:
    norm_tag = normalize_tag(tag)
    entry = get_tag_entry(norm_tag, CATALOG_INDEX_PATH)
    if entry is None:
        raise ValueError(f"tag não catalogada: {norm_tag}")

    paradigm_npz = Path(str(entry.get("paradigm_npz") or "")).resolve()
    if not paradigm_npz.exists():
        raise ValueError(f"paradigma da tag não encontrado: {paradigm_npz}")

    model_name = str(entry.get("model_name") or "").strip()
    if not model_name:
        model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    paradigm, threshold_raw, _src = _load_paradigm(paradigm_npz)
    threshold = threshold_raw
    if threshold_override_pct is not None:
        threshold = float(threshold_override_pct) / 100.0

    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"diretório inválido: {dir_path}")

    pdfs = sorted([p for p in dir_path.glob("*.pdf") if p.is_file()])
    if not pdfs:
        raise ValueError(f"nenhum PDF encontrado em: {dir_path}")

    rows: list[dict[str, Any]] = []
    texts: list[str] = []
    valid_idx: list[int] = []

    for i, pdf in enumerate(pdfs):
        txt = _read_page_text(pdf, int(page_index))
        chars = len(txt)
        row = {
            "file": pdf.name,
            "pdf": str(pdf.resolve()),
            "page_index": int(page_index),
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

    payload = {
        "status": "ok",
        "operation": "find_docs",
        "tag": norm_tag,
        "model_name": model_name,
        "catalog_index": str(CATALOG_INDEX_PATH.resolve()),
        "paradigm_npz": str(paradigm_npz),
        "dir": str(dir_path.resolve()),
        "page_index": int(page_index),
        "threshold_pct": (round(float(threshold) * 100.0, 4) if threshold is not None else None),
        "total_files": len(rows),
        "found_count": len(found_sorted),
        "found": found_sorted,
        "all_results": rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "operation": "find_docs",
        "tag": norm_tag,
        "out": str(out_path.resolve()),
        "total_files": len(rows),
        "found_count": len(found_sorted),
        "threshold_pct": payload["threshold_pct"],
    }, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "find_docs: inferência por TAG já catalogada. "
            "Compara página 0 (ou page-index informado) dos PDFs do diretório com o paradigma da TAG."
        )
    )
    ap.add_argument("--tag", required=True, help="TAG catalogada (ex: despacho)")
    ap.add_argument("--dir", required=True, help="Diretório com PDFs para busca")
    ap.add_argument("--page-index", type=int, default=0, help="Página para comparar (0-based, oficial=0)")
    ap.add_argument("--min-chars", type=int, default=40, help="Mínimo de caracteres para comparar")
    ap.add_argument(
        "--threshold-override-pct",
        type=float,
        default=None,
        help="Override do limiar em percentual (ex: 90.0)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="JSON de saída (padrão: io/catalogacao_classica/outputs/find_docs_<tag>_<ts>.json)",
    )
    args = ap.parse_args(argv)

    norm_tag = normalize_tag(str(args.tag))
    out = (
        Path(args.out)
        if args.out
        else Path("io/catalogacao_classica/outputs") / f"find_docs_{norm_tag}_{_now_compact()}.json"
    )
    return run(
        tag=norm_tag,
        dir_path=Path(args.dir),
        out_path=out,
        page_index=int(args.page_index),
        min_chars=int(args.min_chars),
        threshold_override_pct=(float(args.threshold_override_pct) if args.threshold_override_pct is not None else None),
    )


if __name__ == "__main__":
    raise SystemExit(main())
