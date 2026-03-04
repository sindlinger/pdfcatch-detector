from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import numpy as np

from pdfcatch.family.features import extract_page_text_from_doc

CATALOG_PAGE_INDEX = 0
_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def _normalize_ai_raw_score(raw: float) -> float | None:
    try:
        v = float(raw)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None

    if 0.0 <= v <= 1.0:
        unit = v
    elif -1.0 <= v <= 1.0:
        unit = (v + 1.0) / 2.0
    else:
        unit = 1.0 / (1.0 + math.exp(-v))
    return float(max(0.0, min(1.0, unit)))


def _load_crossencoder_gpu(model_name: str) -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise ValueError(
            "tokenização neural exige GPU: pacote 'torch' não encontrado. "
            "instale torch com CUDA habilitado."
        ) from exc

    if not bool(torch.cuda.is_available()):
        raise ValueError(
            "tokenização neural oficial exige GPU CUDA ativa. "
            "fallback para CPU não é permitido."
        )

    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception as exc:
        raise ValueError(
            "tokenização neural exige pacote 'sentence-transformers'."
        ) from exc

    key = (str(model_name), "cuda")
    model = _MODEL_CACHE.get(key)
    if model is None:
        try:
            model = CrossEncoder(str(model_name), device="cuda")
        except Exception as exc:
            raise ValueError(f"falha ao carregar CrossEncoder em GPU: model={model_name} erro={exc}") from exc
        _MODEL_CACHE[key] = model
    return {"device": "cuda", "model_name": str(model_name), "model": model}


def _read_cover_text(pdf: Path) -> str:
    try:
        doc = fitz.open(pdf)
    except Exception:
        return ""
    try:
        return _normalize_text(extract_page_text_from_doc(doc, CATALOG_PAGE_INDEX))
    finally:
        try:
            doc.close()
        except Exception:
            pass


def _predict_scores(runtime: dict[str, Any], pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
    model = runtime.get("model")
    if model is None:
        return []
    raw = model.predict(
        pairs,
        batch_size=max(1, int(batch_size)),
        show_progress_bar=False,
    )
    if isinstance(raw, np.ndarray):
        raw_vals = [float(x) for x in raw.reshape(-1).tolist()]
    elif isinstance(raw, (list, tuple)):
        raw_vals = [float(x) for x in raw]
    else:
        raw_vals = [float(raw)]

    out: list[float] = []
    for v in raw_vals:
        nv = _normalize_ai_raw_score(v)
        if nv is None:
            out.append(0.0)
        else:
            out.append(float(nv))
    return out


def _build_pairwise_matrix(
    runtime: dict[str, Any],
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    n = len(texts)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        pairs = [(texts[i], texts[j]) for j in range(n)]
        vals = _predict_scores(runtime, pairs, batch_size)
        if len(vals) != n:
            raise ValueError(f"pontuação inconsistente no par i={i}: esperado={n} recebido={len(vals)}")
        mat[i, :] = np.array(vals, dtype=np.float32)
    return mat


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
    dir_path: Path,
    paradigm_out: Path,
    report_out: Path,
    model_name: str,
    batch_size: int,
    min_chars: int,
    prototype_count: int,
    threshold_method: str,
    threshold_sigma: float,
    max_files: int,
) -> int:
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"diretório inválido: {dir_path}")

    pdfs = sorted(p for p in dir_path.glob("*.pdf") if p.is_file())
    if max_files > 0:
        pdfs = pdfs[: int(max_files)]
    if not pdfs:
        raise ValueError(f"nenhum PDF encontrado em: {dir_path}")

    files: list[str] = []
    texts: list[str] = []
    skipped: list[dict[str, Any]] = []
    for pdf in pdfs:
        txt = _read_cover_text(pdf)
        if len(txt) < int(min_chars):
            skipped.append({"file": pdf.name, "reason": f"cover_text_too_short({len(txt)}<{int(min_chars)})"})
            continue
        files.append(pdf.name)
        texts.append(txt)

    if len(texts) < 3:
        raise ValueError(
            "capas válidas insuficientes para paradigma neural. "
            f"necessário >=3; recebido={len(texts)}"
        )

    runtime = _load_crossencoder_gpu(str(model_name))
    pairwise = _build_pairwise_matrix(runtime, texts, batch_size=int(batch_size))

    n = pairwise.shape[0]
    if n != len(files):
        raise ValueError("inconsistência interna no tamanho da matriz de similaridade")

    mean_sim_by_row = []
    for i in range(n):
        # Exclui auto-comparação para evitar viés.
        vals = [float(pairwise[i, j]) for j in range(n) if j != i]
        mean_sim_by_row.append(float(sum(vals) / max(1, len(vals))))
    medoid_idx = int(np.argmax(np.array(mean_sim_by_row, dtype=np.float32)))

    # Paradigma: protótipos mais próximos do medóide via similaridade neural.
    pcount = max(1, min(int(prototype_count), n))
    sims_to_medoid = [(j, float(pairwise[medoid_idx, j])) for j in range(n)]
    sims_to_medoid.sort(key=lambda x: x[1], reverse=True)
    proto_indices = [idx for idx, _ in sims_to_medoid[:pcount]]
    proto_texts = [texts[i] for i in proto_indices]

    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for i in range(n):
        pairs = [(texts[i], ptxt) for ptxt in proto_texts]
        vals = _predict_scores(runtime, pairs, batch_size=int(batch_size))
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float32)
        max_score = float(np.max(arr))
        mean_score = float(np.mean(arr))
        # Peso maior no máximo, mantendo contribuição do conjunto de protótipos.
        final_unit = float((0.7 * max_score) + (0.3 * mean_score))
        final_pct = float(round(final_unit * 100.0, 4))
        scores.append(final_pct)
        rows.append(
            {
                "file": files[i],
                "similarity_pct": final_pct,
                "max_score_pct": float(round(max_score * 100.0, 4)),
                "mean_score_pct": float(round(mean_score * 100.0, 4)),
                "prototype_count": int(len(proto_texts)),
                "medoid_similarity_pct": float(round(float(pairwise[medoid_idx, i]) * 100.0, 4)),
            }
        )

    if not scores:
        raise ValueError("falha na validação neural: nenhuma pontuação válida foi gerada")

    score_arr = np.array(scores, dtype=np.float64)
    threshold_pct, threshold_meta = _threshold_from_scores(
        score_arr, str(threshold_method), float(threshold_sigma)
    )
    for row in rows:
        row["valid_by_threshold"] = bool(float(row["similarity_pct"]) >= float(threshold_pct))
    rows.sort(key=lambda r: float(r["similarity_pct"]))

    valid_count = int(sum(1 for r in rows if bool(r.get("valid_by_threshold"))))
    valid_ratio = float(valid_count / max(1, len(rows)))
    status = _status_from_ratio(valid_ratio)

    paradigm_payload = {
        "schema": "token_paradigm.v2.crossencoder_gpu",
        "created_at_utc": _now_utc_iso(),
        "source_dir": str(dir_path.resolve()),
        "catalog_page_index": CATALOG_PAGE_INDEX,
        "model_name": str(runtime.get("model_name")),
        "device": str(runtime.get("device")),
        "medoid_index": int(medoid_idx),
        "medoid_file": files[medoid_idx],
        "prototype_indices": [int(i) for i in proto_indices],
        "prototype_files": [files[i] for i in proto_indices],
        "prototype_count": int(len(proto_indices)),
        "selection_mode": "crossencoder_medoid_topk",
    }
    paradigm_out.parent.mkdir(parents=True, exist_ok=True)
    paradigm_out.write_text(json.dumps(paradigm_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_payload = {
        "schema": "token_paradigm_validation.v2.crossencoder_gpu",
        "created_at_utc": _now_utc_iso(),
        "status": status,
        "objective": "validar gabarito positivo por tokenização neural (CrossEncoder GPU) capa_vs_paradigma",
        "comparison_mode": "cover_vs_neural_token_paradigm_only",
        "catalog_page_index": CATALOG_PAGE_INDEX,
        "model_name": str(runtime.get("model_name")),
        "device": str(runtime.get("device")),
        "input_dir": str(dir_path.resolve()),
        "output_paradigm_json": str(paradigm_out.resolve()),
        "threshold": {
            "value_pct": float(round(float(threshold_pct), 4)),
            **threshold_meta,
        },
        "metrics": {
            "total_files": int(len(pdfs)),
            "files_validated": int(len(rows)),
            "files_skipped": int(len(skipped)),
            "valid_count": valid_count,
            "invalid_count": int(len(rows) - valid_count),
            "valid_ratio": float(round(valid_ratio, 6)),
            "similarity_pct_min": float(round(float(np.min(score_arr)), 4)),
            "similarity_pct_p10": float(round(float(_safe_percentile(score_arr, 10.0)), 4)),
            "similarity_pct_median": float(round(float(np.median(score_arr)), 4)),
            "similarity_pct_mean": float(round(float(np.mean(score_arr)), 4)),
            "similarity_pct_p90": float(round(float(_safe_percentile(score_arr, 90.0)), 4)),
            "similarity_pct_max": float(round(float(np.max(score_arr)), 4)),
        },
        "lowest_similarity_covers": rows[: min(20, len(rows))],
        "skipped": skipped[:200],
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "status": "ok",
        "validation_status": status,
        "output_paradigm_json": str(paradigm_out.resolve()),
        "output_report_json": str(report_out.resolve()),
        "files_validated": int(len(rows)),
        "files_skipped": int(len(skipped)),
        "valid_count": valid_count,
        "invalid_count": int(len(rows) - valid_count),
        "valid_ratio": float(round(valid_ratio, 6)),
        "threshold_pct": float(round(float(threshold_pct), 4)),
        "medoid_file": files[medoid_idx],
        "prototype_count": int(len(proto_indices)),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Token Paradigm Validate (alta precisão): cria paradigma por tokenização neural "
            "com CrossEncoder em GPU e valida capas (página 0) contra esse paradigma."
        )
    )
    ap.add_argument("--dir", required=True, help="Diretório com PDFs de despachos validados")
    ap.add_argument(
        "--paradigm-out",
        default="io/out/despacho_token_paradigm.json",
        help="Saída JSON com o paradigma neural de tokens",
    )
    ap.add_argument(
        "--report-out",
        default="io/out/despacho_token_validation.json",
        help="Relatório JSON da validação lexical neural",
    )
    ap.add_argument(
        "--model",
        default="BAAI/bge-reranker-v2-m3",
        help="Modelo CrossEncoder (tokenização neural em GPU)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size para predição no CrossEncoder",
    )
    ap.add_argument(
        "--min-chars",
        type=int,
        default=40,
        help="Mínimo de caracteres de capa para participar",
    )
    ap.add_argument(
        "--prototype-count",
        type=int,
        default=8,
        help="Quantidade de protótipos do paradigma (top-k próximos do medóide)",
    )
    ap.add_argument(
        "--threshold-method",
        choices=("p05", "mean_std"),
        default="p05",
        help="Método para derivar limiar de validade",
    )
    ap.add_argument(
        "--threshold-sigma",
        type=float,
        default=2.0,
        help="Sigma usado em threshold-method=mean_std",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limite de arquivos (0 = sem limite)",
    )
    args = ap.parse_args(argv)
    return run(
        dir_path=Path(args.dir),
        paradigm_out=Path(args.paradigm_out),
        report_out=Path(args.report_out),
        model_name=str(args.model),
        batch_size=int(args.batch_size),
        min_chars=int(args.min_chars),
        prototype_count=int(args.prototype_count),
        threshold_method=str(args.threshold_method),
        threshold_sigma=float(args.threshold_sigma),
        max_files=int(args.max_files),
    )


if __name__ == "__main__":
    raise SystemExit(main())
