from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _as_float_matrix(arr: Any) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"embeddings inválido: esperado matriz 2D, recebido shape={tuple(out.shape)}")
    if out.shape[0] < 2:
        raise ValueError("embeddings insuficientes: precisa de pelo menos 2 capas para validar o paradigma")
    if out.shape[1] < 8:
        raise ValueError(f"dimensão de embedding inesperada: {out.shape[1]}")
    return out


def _row_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return x / norms


def _safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def _load_embeddings(npz_path: Path) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if not npz_path.exists() or not npz_path.is_file():
        raise ValueError(f"arquivo de embeddings não encontrado: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        if "embeddings" not in data:
            raise ValueError(f"npz inválido: campo 'embeddings' ausente em {npz_path}")
        emb = _as_float_matrix(data["embeddings"])
        files_raw = data["files"] if "files" in data else np.array([], dtype=object)
        files = [str(x) for x in files_raw.tolist()]
        if len(files) != emb.shape[0]:
            # Mantém alinhamento mesmo com metadado incompleto.
            files = [f"item_{i:06d}" for i in range(emb.shape[0])]

        model_name = None
        source_dir = None
        created_at_utc = None
        if "model_name" in data and len(data["model_name"]) > 0:
            model_name = str(data["model_name"][0])
        if "source_dir" in data and len(data["source_dir"]) > 0:
            source_dir = str(data["source_dir"][0])
        if "created_at_utc" in data and len(data["created_at_utc"]) > 0:
            created_at_utc = str(data["created_at_utc"][0])

    meta = {
        "model_name": model_name,
        "source_dir": source_dir,
        "created_at_utc": created_at_utc,
    }
    return emb, files, meta


def _build_paradigm(emb: np.ndarray) -> np.ndarray:
    centered = _row_normalize(emb)
    paradigm = np.mean(centered, axis=0, dtype=np.float64)
    norm = float(np.linalg.norm(paradigm))
    if norm <= 0.0:
        raise ValueError("falha ao construir paradigma: norma zero")
    return (paradigm / norm).astype(np.float32)


def _threshold_from_scores(scores: np.ndarray, method: str, sigma: float) -> tuple[float, dict[str, Any]]:
    if method == "p05":
        thr = _safe_percentile(scores, 5.0)
        return float(thr), {"method": "p05", "percentile": 5.0}

    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=0))
    thr = float(mean - float(sigma) * std)
    return thr, {"method": "mean_std", "mean": mean, "std": std, "sigma": float(sigma)}


def _status_from_ratio(valid_ratio: float) -> str:
    if valid_ratio >= 0.95:
        return "ok"
    if valid_ratio >= 0.85:
        return "warning"
    return "poor"


def run(
    *,
    embeddings_npz: Path,
    paradigm_out: Path,
    report_out: Path,
    threshold_method: str,
    threshold_sigma: float,
) -> int:
    emb, files, emb_meta = _load_embeddings(embeddings_npz)
    emb = _row_normalize(emb)

    paradigm = _build_paradigm(emb)
    scores = np.clip(np.dot(emb, paradigm), -1.0, 1.0).astype(np.float32)
    scores_pct = (scores * 100.0).astype(np.float32)

    threshold, threshold_meta = _threshold_from_scores(scores.astype(np.float64), threshold_method, threshold_sigma)
    valid_mask = scores >= float(threshold)
    valid_ratio = float(np.mean(valid_mask.astype(np.float32)))
    status = _status_from_ratio(valid_ratio)

    paradigm_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paradigm_out,
        paradigm_embedding=paradigm,
        source_embeddings=np.array([str(embeddings_npz.resolve())], dtype=object),
        created_at_utc=np.array([_now_utc_iso()], dtype=object),
        threshold=np.array([float(threshold)], dtype=np.float32),
        threshold_method=np.array([str(threshold_method)], dtype=object),
    )

    rows: list[dict[str, Any]] = []
    for i, fname in enumerate(files):
        rows.append(
            {
                "file": fname,
                "similarity": float(scores[i]),
                "similarity_pct": float(round(float(scores_pct[i]), 4)),
                "valid_by_threshold": bool(valid_mask[i]),
            }
        )
    rows.sort(key=lambda r: float(r["similarity"]))

    summary = {
        "schema": "ai_paradigm_validation.v1",
        "created_at_utc": _now_utc_iso(),
        "status": status,
        "objective": "validar gabarito positivo (despacho) por similaridade capa_vs_paradigma",
        "comparison_mode": "cover_vs_paradigm_only",
        "input_embeddings_npz": str(embeddings_npz.resolve()),
        "output_paradigm_npz": str(paradigm_out.resolve()),
        "input_meta": emb_meta,
        "threshold": {
            "value": float(threshold),
            "value_pct": float(round(float(threshold * 100.0), 4)),
            **threshold_meta,
        },
        "metrics": {
            "total_covers": int(scores.size),
            "valid_count": int(np.sum(valid_mask)),
            "invalid_count": int(scores.size - int(np.sum(valid_mask))),
            "valid_ratio": float(round(valid_ratio, 6)),
            "similarity_pct_min": float(round(float(np.min(scores_pct)), 4)),
            "similarity_pct_p10": float(round(float(_safe_percentile(scores_pct, 10.0)), 4)),
            "similarity_pct_median": float(round(float(np.median(scores_pct)), 4)),
            "similarity_pct_mean": float(round(float(np.mean(scores_pct)), 4)),
            "similarity_pct_p90": float(round(float(_safe_percentile(scores_pct, 90.0)), 4)),
            "similarity_pct_max": float(round(float(np.max(scores_pct)), 4)),
        },
        "lowest_similarity_covers": rows[: min(20, len(rows))],
    }

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "status": "ok",
        "validation_status": status,
        "output_paradigm_npz": str(paradigm_out.resolve()),
        "output_report_json": str(report_out.resolve()),
        "total_covers": int(scores.size),
        "valid_count": int(np.sum(valid_mask)),
        "invalid_count": int(scores.size - int(np.sum(valid_mask))),
        "valid_ratio": float(round(valid_ratio, 6)),
        "threshold_pct": float(round(float(threshold * 100.0), 4)),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "AI Paradigm Validate: gera embedding paradigma a partir das capas "
            "de despacho e valida cada capa contra esse paradigma (sem comparar capa-vs-capa)."
        )
    )
    ap.add_argument(
        "--embeddings",
        required=True,
        help="Arquivo .npz gerado pelo ai_embed_covers",
    )
    ap.add_argument(
        "--paradigm-out",
        default="io/catalogacao_classica/despacho_paradigm_embedding.npz",
        help="Saída .npz do embedding paradigma",
    )
    ap.add_argument(
        "--report-out",
        default="io/catalogacao_classica/despacho_paradigm_validation.json",
        help="Relatório JSON da validação capa_vs_paradigma",
    )
    ap.add_argument(
        "--threshold-method",
        choices=("p05", "mean_std"),
        default="p05",
        help="Método para derivar limiar de validade a partir do próprio gabarito",
    )
    ap.add_argument(
        "--threshold-sigma",
        type=float,
        default=2.0,
        help="Sigma usado em threshold-method=mean_std",
    )
    args = ap.parse_args(argv)
    return run(
        embeddings_npz=Path(args.embeddings),
        paradigm_out=Path(args.paradigm_out),
        report_out=Path(args.report_out),
        threshold_method=str(args.threshold_method),
        threshold_sigma=float(args.threshold_sigma),
    )


if __name__ == "__main__":
    raise SystemExit(main())
