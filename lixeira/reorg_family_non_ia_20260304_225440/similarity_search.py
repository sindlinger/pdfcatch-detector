from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pdfcatch.family.catalog_registry import DEFAULT_REGISTRY_PATH, normalize_tag, resolve_catalog_by_tag
from pdfcatch.family.similarity_score import (
    AI_BATCH_SIZE_DEFAULT,
    AI_MAX_EXAMPLES_DEFAULT,
    AI_MODEL_DEFAULT,
    score_pdf_against_catalog,
)


def _now_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug(name: str) -> str:
    out = []
    for ch in str(name):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "arquivo"


def _base_out_dir(tag: str) -> Path:
    return Path("io/out") / f"similarity_search_tag_{normalize_tag(tag)}_{_now_compact()}"


def _annotate_result(res: dict[str, Any], *, tag: str, ref_path: Path, ref: dict[str, Any]) -> dict[str, Any]:
    out = dict(res)
    out["requested_tag"] = normalize_tag(tag)
    out["catalog_path"] = str(ref_path)
    out["catalog_tag"] = ref.get("catalog_tag")
    return out


def run_pdf(
    *,
    pdf: Path,
    tag: str,
    registry_path: Path,
    out: Path | None,
    ai_model: str,
    ai_batch_size: int,
    ai_max_examples: int,
) -> int:
    norm_tag = normalize_tag(tag)
    ref_path = resolve_catalog_by_tag(registry_path, norm_tag)
    ref = json.loads(ref_path.read_text(encoding="utf-8"))

    res = score_pdf_against_catalog(
        pdf,
        ref,
        ai_model=str(ai_model),
        ai_batch_size=int(ai_batch_size),
        ai_max_examples=int(ai_max_examples),
    )
    res = _annotate_result(res, tag=norm_tag, ref_path=ref_path, ref=ref)

    if out is None:
        out_dir = _base_out_dir(norm_tag)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"similarity_search__{_slug(pdf.stem)}.json"
    else:
        out_path = out
        out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {
        "status": "ok",
        "saved_path": str(out_path.resolve()),
        "pdf": str(pdf),
        "tag": norm_tag,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_dir(
    *,
    dir_path: Path,
    tag: str,
    registry_path: Path,
    out_dir: Path | None,
    ai_model: str,
    ai_batch_size: int,
    ai_max_examples: int,
) -> int:
    norm_tag = normalize_tag(tag)
    ref_path = resolve_catalog_by_tag(registry_path, norm_tag)
    ref = json.loads(ref_path.read_text(encoding="utf-8"))

    pdfs = sorted(p for p in dir_path.glob("*.pdf") if p.is_file())
    if not pdfs:
        raise ValueError(f"nenhum PDF encontrado em: {dir_path}")

    target_dir = out_dir or _base_out_dir(norm_tag)
    target_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for pdf in pdfs:
        res = score_pdf_against_catalog(
            pdf,
            ref,
            ai_model=str(ai_model),
            ai_batch_size=int(ai_batch_size),
            ai_max_examples=int(ai_max_examples),
        )
        rows.append(_annotate_result(res, tag=norm_tag, ref_path=ref_path, ref=ref))

    detected_count = int(sum(1 for r in rows if bool(r.get("detected"))))
    summary = {
        "timestamp_utc": _now_utc_iso(),
        "tag": norm_tag,
        "catalog_path": str(ref_path),
        "input_dir": str(dir_path.resolve()),
        "total_files": int(len(rows)),
        "detected_count": detected_count,
        "not_detected_count": int(len(rows) - detected_count),
        "detected_ratio": float(detected_count / len(rows)) if rows else 0.0,
    }

    results_path = target_dir / "similarity_search_results.json"
    summary_path = target_dir / "similarity_search_summary.json"
    results_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "status": "ok",
        "saved_results": str(results_path.resolve()),
        "saved_summary": str(summary_path.resolve()),
        "total_files": summary["total_files"],
        "detected_count": summary["detected_count"],
        "not_detected_count": summary["not_detected_count"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Similarity Search: usa catálogo já registrado para buscar semelhança em PDF único ou diretório (IA lexical em GPU obrigatória)"
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", help="PDF único para inferência")
    group.add_argument("--dir", help="Diretório com PDFs para inferência em lote")
    ap.add_argument("--tag", required=True, help="TAG já catalogada (ex.: despacho)")
    ap.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY_PATH),
        help="Registry oficial de tags catalogadas",
    )
    ap.add_argument(
        "--out",
        default=None,
        help=(
            "Saída: para --pdf é arquivo .json; para --dir é diretório de saída. "
            "se omitido, gera automaticamente em io/out/"
        ),
    )
    ap.add_argument(
        "--ai-model",
        default=AI_MODEL_DEFAULT,
        help="Modelo CrossEncoder para IA lexical (GPU obrigatória)",
    )
    ap.add_argument(
        "--ai-batch-size",
        type=int,
        default=AI_BATCH_SIZE_DEFAULT,
        help="Batch da IA lexical em GPU",
    )
    ap.add_argument(
        "--ai-max-examples",
        type=int,
        default=AI_MAX_EXAMPLES_DEFAULT,
        help="Máximo de textos de referência por família na IA lexical",
    )
    args = ap.parse_args(argv)

    if args.pdf:
        return run_pdf(
            pdf=Path(args.pdf),
            tag=str(args.tag),
            registry_path=Path(args.registry),
            out=Path(args.out) if args.out else None,
            ai_model=str(args.ai_model),
            ai_batch_size=int(args.ai_batch_size),
            ai_max_examples=int(args.ai_max_examples),
        )
    return run_dir(
        dir_path=Path(args.dir),
        tag=str(args.tag),
        registry_path=Path(args.registry),
        out_dir=Path(args.out) if args.out else None,
        ai_model=str(args.ai_model),
        ai_batch_size=int(args.ai_batch_size),
        ai_max_examples=int(args.ai_max_examples),
    )


if __name__ == "__main__":
    raise SystemExit(main())
