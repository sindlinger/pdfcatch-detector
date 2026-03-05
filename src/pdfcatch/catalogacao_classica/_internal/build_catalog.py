from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pdfcatch.catalogacao_classica._internal.ai_embed_covers import run as run_embed
from pdfcatch.catalogacao_classica._internal.ai_paradigm_validate import run as run_paradigm
from pdfcatch.catalogacao_classica._internal.catalog_store import (
    CATALOG_INDEX_PATH,
    catalog_dir_for_tag,
    normalize_tag,
    now_utc_iso,
    set_tag_entry,
    get_tag_entry,
)


def run(
    *,
    tag: str,
    dir_path: Path,
    update: bool,
    model_name: str,
    batch_size: int,
    min_chars: int,
    max_files: int,
    threshold_method: str,
    threshold_sigma: float,
) -> int:
    norm_tag = normalize_tag(tag)
    existing = get_tag_entry(norm_tag, CATALOG_INDEX_PATH)
    if existing is not None and not bool(update):
        raise ValueError(
            f"tag já catalogada: '{norm_tag}'. "
            "use --update para atualizar explicitamente."
        )

    cdir = catalog_dir_for_tag(norm_tag)
    cdir.mkdir(parents=True, exist_ok=True)

    embeddings_npz = cdir / "cover_embeddings.npz"
    paradigm_npz = cdir / "paradigm_embedding.npz"
    paradigm_report = cdir / "paradigm_validation.json"
    catalog_json = cdir / "catalog.json"

    rc1 = run_embed(
        dir_path=dir_path,
        out_path=embeddings_npz,
        model_name=str(model_name),
        batch_size=int(batch_size),
        min_chars=int(min_chars),
        max_files=int(max_files),
    )
    if rc1 != 0:
        raise RuntimeError(f"falha no build de embeddings: rc={rc1}")

    rc2 = run_paradigm(
        embeddings_npz=embeddings_npz,
        paradigm_out=paradigm_npz,
        report_out=paradigm_report,
        threshold_method=str(threshold_method),
        threshold_sigma=float(threshold_sigma),
    )
    if rc2 != 0:
        raise RuntimeError(f"falha no build de paradigma: rc={rc2}")

    report_obj: dict[str, Any] = json.loads(paradigm_report.read_text(encoding="utf-8"))
    threshold_pct = ((report_obj.get("threshold") or {}).get("value_pct"))

    meta_obj = json.loads((embeddings_npz.with_suffix(embeddings_npz.suffix + ".meta.json")).read_text(encoding="utf-8"))

    catalog_obj = {
        "schema": "tag_catalog.v1",
        "created_at_utc": now_utc_iso(),
        "tag": norm_tag,
        "source_dir": str(dir_path.resolve()),
        "model_name": str(model_name),
        "catalog_page_index": 0,
        "embeddings_npz": str(embeddings_npz.resolve()),
        "paradigm_npz": str(paradigm_npz.resolve()),
        "paradigm_report_json": str(paradigm_report.resolve()),
        "threshold_pct": threshold_pct,
        "files_total": meta_obj.get("files_total"),
        "files_embedded": meta_obj.get("files_embedded"),
        "files_skipped": meta_obj.get("files_skipped"),
    }
    catalog_json.write_text(json.dumps(catalog_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    set_tag_entry(
        norm_tag,
        {
            "catalog_json": str(catalog_json.resolve()),
            "paradigm_npz": str(paradigm_npz.resolve()),
            "paradigm_report_json": str(paradigm_report.resolve()),
            "embeddings_npz": str(embeddings_npz.resolve()),
            "model_name": str(model_name),
            "catalog_page_index": 0,
            "updated_at_utc": now_utc_iso(),
        },
        CATALOG_INDEX_PATH,
    )

    payload = {
        "status": "ok",
        "operation": "build_catalog",
        "tag": norm_tag,
        "catalog_json": str(catalog_json.resolve()),
        "embeddings_npz": str(embeddings_npz.resolve()),
        "paradigm_npz": str(paradigm_npz.resolve()),
        "paradigm_report_json": str(paradigm_report.resolve()),
        "catalog_index": str(CATALOG_INDEX_PATH.resolve()),
        "threshold_pct": threshold_pct,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "build_catalog: gera catálogo de uma TAG (embeddings da capa página 0 + paradigma). "
            "Bloqueia recatalogação por padrão."
        )
    )
    ap.add_argument("--tag", required=True, help="TAG do catálogo (ex: despacho)")
    ap.add_argument("--dir", required=True, help="Diretório com PDFs da tag")
    ap.add_argument("--update", action="store_true", help="Atualiza TAG já catalogada")
    ap.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="Modelo de embedding",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--threshold-method", choices=("p05", "mean_std"), default="p05")
    ap.add_argument("--threshold-sigma", type=float, default=2.0)
    args = ap.parse_args(argv)
    return run(
        tag=str(args.tag),
        dir_path=Path(args.dir),
        update=bool(args.update),
        model_name=str(args.model),
        batch_size=int(args.batch_size),
        min_chars=int(args.min_chars),
        max_files=int(args.max_files),
        threshold_method=str(args.threshold_method),
        threshold_sigma=float(args.threshold_sigma),
    )


if __name__ == "__main__":
    raise SystemExit(main())
