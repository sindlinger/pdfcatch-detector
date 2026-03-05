from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pdfcatch.catalogacao_classica._internal.ai_embed_covers import run as run_embed
from pdfcatch.catalogacao_classica._internal.ai_paradigm_validate import run as run_paradigm
from pdfcatch.qwen_supervisionado._internal.catalog_store import (
    CATALOG_INDEX_PATH,
    catalog_dir_for_tag,
    normalize_tag,
    now_utc_iso,
    set_tag_entry,
    get_tag_entry,
)
from pdfcatch.qwen_supervisionado._internal.paths import QWEN_INPUT_ROOT, QWEN_CATALOG_ROOT, bootstrap_layout, ensure_input_dir


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
    bootstrap_layout()
    source_dir = ensure_input_dir(dir_path)
    norm_tag = normalize_tag(tag)
    existing = get_tag_entry(norm_tag, CATALOG_INDEX_PATH)
    if existing is not None and not bool(update):
        raise ValueError(
            f"tag ja catalogada: '{norm_tag}'. "
            "use --update para atualizar explicitamente."
        )

    cdir = catalog_dir_for_tag(norm_tag)

    embeddings_npz = cdir / "cover_embeddings.npz"
    paradigm_npz = cdir / "paradigm_embedding.npz"
    paradigm_report = cdir / "paradigm_validation.json"
    catalog_json = cdir / "catalog.json"

    rc1 = run_embed(
        dir_path=source_dir,
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
        "scope": "qwen_isolated",
        "created_at_utc": now_utc_iso(),
        "tag": norm_tag,
        "source_dir": str(source_dir),
        "input_root": str(QWEN_INPUT_ROOT.resolve()),
        "catalog_root": str(QWEN_CATALOG_ROOT.resolve()),
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
            "scope": "qwen_isolated",
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
        "scope": "qwen_isolated",
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
            "build_catalog (Qwen isolado): gera catalogo de TAG (pagina 0) "
            "e grava apenas no sandbox io/qwen_isolated."
        )
    )
    ap.add_argument("--tag", required=True, help="TAG do catalogo (ex: despacho)")
    ap.add_argument(
        "--dir",
        required=True,
        help="Diretorio de entrada dentro de io/qwen_isolated/input",
    )
    ap.add_argument("--update", action="store_true", help="Atualiza TAG ja catalogada")
    ap.add_argument(
        "--model",
        required=True,
        help="Modelo de embedding (Qwen ou outro informado explicitamente)",
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

