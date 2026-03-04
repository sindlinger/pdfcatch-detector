from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import fitz
import numpy as np

from pdfcatch.family.catalog_registry import DEFAULT_REGISTRY_PATH, normalize_tag, resolve_catalog_by_tag
from pdfcatch.family.features import (
    FEATURE_FIELDS,
    extract_page_features_from_doc,
    extract_page_text_from_doc,
    extract_page_tokens_from_doc,
)

CATALOG_PAGE_INDEX = 0
POSITIVE_CONFIDENCE = {"high", "medium"}
TOKEN_PAGE_MAX_TERMS = 512
AI_MODEL_DEFAULT = "BAAI/bge-reranker-v2-m3"
AI_BATCH_SIZE_DEFAULT = 16
AI_MAX_EXAMPLES_DEFAULT = 4
_AI_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _empty_result(page_idx: int, *, confidence: str, status: str) -> dict[str, Any]:
    return {
        "page_index": page_idx,
        "catalog_page_index": CATALOG_PAGE_INDEX,
        "mode": None,
        "family_id": None,
        "cluster": None,
        "family": None,
        "distance": None,
        "scores": None,
        "confidence": confidence,
        "family_status": status,
        "thresholds": None,
        "rule_applied": None,
        "selection_strategy": None,
        "family_similarity_pct": None,
        "token_similarity_pct": None,
        "token_stage": None,
        "ai_similarity_pct": None,
        "ai_stage": None,
        "feature_similarity_pct": None,
        "feature_delta_z": None,
        "family_comparison": [],
    }


def _is_positive_confidence(confidence: str | None) -> bool:
    return str(confidence or "").strip().lower() in POSITIVE_CONFIDENCE


def _confidence_rank(confidence: str | None) -> int:
    c = str(confidence or "").strip().lower()
    if c == "high":
        return 3
    if c == "medium":
        return 2
    if c == "unknown":
        return 1
    return 0


def _entry_rows_from_hits(hit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Mantém só páginas de entrada de bloco contínuo de hits.
    Ex.: hits em 10,11,12,20 -> entradas 10 e 20.
    """
    ordered = sorted(
        [r for r in hit_rows if isinstance(r.get("page_index"), int)],
        key=lambda r: int(r["page_index"]),
    )
    out: list[dict[str, Any]] = []
    prev_idx: int | None = None
    for row in ordered:
        idx = int(row["page_index"])
        if prev_idx is None or idx != prev_idx + 1:
            out.append(row)
        prev_idx = idx
    return out


def _family_meta_from_page(page_ref: dict[str, Any], family_id: int) -> dict[str, Any] | None:
    families = page_ref.get("families")
    if isinstance(families, dict):
        meta = families.get(str(int(family_id)))
        if isinstance(meta, dict):
            return meta
    registry = page_ref.get("family_registry") or []
    if isinstance(registry, list):
        for row in registry:
            if int(row.get("family_id", -1)) == int(family_id):
                return {
                    "dist_mean": row.get("family_dist_mean"),
                    "dist_max": row.get("family_dist_max"),
                    "family_token_model": row.get("family_token_model"),
                    "family_reference_examples": row.get("family_reference_examples"),
                }
    return None


def _resolve_family_confidence(
    distance: float, family_meta: dict[str, Any] | None
) -> tuple[str, str, dict[str, float | None], str]:
    dist_mean = _safe_float((family_meta or {}).get("dist_mean"))
    dist_max = _safe_float((family_meta or {}).get("dist_max"))
    thresholds = {"dist_mean": dist_mean, "dist_max": dist_max}
    rule = "distance <= dist_mean -> high; dist_mean < distance <= dist_max -> medium; distance > dist_max -> low_out_of_family"
    if dist_mean is None or dist_max is None or dist_mean > dist_max:
        return "unknown", "unknown", thresholds, rule
    if distance <= dist_mean:
        return "high", "in_family", thresholds, rule
    if distance <= dist_max:
        return "medium", "in_family", thresholds, rule
    return "low_out_of_family", "outside_family", thresholds, rule


def _feature_similarity_pct(delta_z: float) -> float:
    # 100% quando igual ao centróide; cai suavemente conforme aumenta o delta normalizado.
    return float(max(0.0, min(100.0, 100.0 * np.exp(-0.5 * (delta_z**2)))))


def _tokens_to_weights(tokens: list[str], max_terms: int) -> dict[str, float]:
    counter: Counter[str] = Counter(tokens or [])
    total = int(sum(counter.values()))
    if total <= 0:
        return {}
    out: dict[str, float] = {}
    for token, count in counter.most_common(max_terms):
        out[str(token)] = float(int(count) / total)
    return out


def _family_token_weights(family_meta: dict[str, Any] | None) -> dict[str, float]:
    model = (family_meta or {}).get("family_token_model")
    if not isinstance(model, dict):
        model = (family_meta or {}).get("token_model")
    if not isinstance(model, dict):
        return {}
    terms = model.get("terms")
    if not isinstance(terms, list):
        return {}

    out: dict[str, float] = {}
    total_from_count = 0.0
    count_map: dict[str, float] = {}
    for row in terms:
        if not isinstance(row, dict):
            continue
        token = str(row.get("token") or "").strip()
        if not token:
            continue
        weight = _safe_float(row.get("weight"))
        if weight is not None and weight > 0:
            out[token] = float(weight)
            continue
        count = _safe_float(row.get("count"))
        if count is not None and count > 0:
            count_map[token] = float(count)
            total_from_count += float(count)

    if out:
        norm = float(sum(out.values()))
        if norm > 0:
            return {k: float(v / norm) for k, v in out.items()}
        return {}
    if total_from_count > 0:
        return {k: float(v / total_from_count) for k, v in count_map.items()}
    return {}


def _token_similarity_stage(page_weights: dict[str, float], family_weights: dict[str, float]) -> dict[str, Any]:
    if not page_weights or not family_weights:
        return {
            "status": "unavailable",
            "token_similarity_pct": None,
            "weighted_jaccard": None,
            "cosine": None,
            "shared_terms": 0,
            "page_terms": int(len(page_weights)),
            "family_terms": int(len(family_weights)),
        }

    keys = set(page_weights.keys()) | set(family_weights.keys())
    num = 0.0
    den = 0.0
    dot = 0.0
    sum_page_sq = 0.0
    sum_family_sq = 0.0
    shared_terms = 0
    for k in keys:
        pv = float(page_weights.get(k, 0.0))
        fv = float(family_weights.get(k, 0.0))
        num += min(pv, fv)
        den += max(pv, fv)
        dot += pv * fv
        sum_page_sq += pv * pv
        sum_family_sq += fv * fv
        if pv > 0.0 and fv > 0.0:
            shared_terms += 1

    weighted_jaccard = float(num / den) if den > 0 else 0.0
    cosine = float(dot / ((sum_page_sq ** 0.5) * (sum_family_sq ** 0.5))) if sum_page_sq > 0 and sum_family_sq > 0 else 0.0
    token_similarity_pct = float(round((0.5 * weighted_jaccard + 0.5 * cosine) * 100.0, 2))
    return {
        "status": "ok",
        "token_similarity_pct": token_similarity_pct,
        "weighted_jaccard": float(round(weighted_jaccard, 6)),
        "cosine": float(round(cosine, 6)),
        "shared_terms": int(shared_terms),
        "page_terms": int(len(page_weights)),
        "family_terms": int(len(family_weights)),
    }


def _require_ai_gpu_runtime(model_name: str) -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise ValueError(
            "IA lexical com GPU obrigatória: pacote 'torch' não encontrado. "
            "instale torch com CUDA habilitado."
        ) from exc

    if not bool(torch.cuda.is_available()):
        raise ValueError(
            "IA lexical oficial exige GPU CUDA ativa. "
            "fallback para CPU foi removido por política."
        )

    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception as exc:
        raise ValueError(
            "IA lexical com GPU obrigatória: pacote 'sentence-transformers' não encontrado."
        ) from exc

    device = "cuda"
    key = (str(model_name), device)
    model = _AI_MODEL_CACHE.get(key)
    if model is None:
        try:
            model = CrossEncoder(str(model_name), device=device)
        except Exception as exc:
            raise ValueError(
                "falha ao carregar modelo de IA lexical em GPU. "
                f"model={model_name} erro={exc}"
            ) from exc
        _AI_MODEL_CACHE[key] = model

    return {
        "enabled": True,
        "device": device,
        "model_name": str(model_name),
        "model": model,
    }


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


def _family_reference_texts(family_meta: dict[str, Any] | None, max_examples: int) -> list[str]:
    if max_examples <= 0:
        return []

    texts: list[str] = []

    token_model = (family_meta or {}).get("family_token_model")
    if not isinstance(token_model, dict):
        token_model = (family_meta or {}).get("token_model")
    if isinstance(token_model, dict):
        refs = token_model.get("reference_texts")
        if isinstance(refs, list):
            for t in refs:
                s = str(t or "").strip()
                if s:
                    texts.append(s)

    for key in ("family_reference_examples", "examples"):
        arr = (family_meta or {}).get(key)
        if not isinstance(arr, list):
            continue
        for row in arr:
            if not isinstance(row, dict):
                continue
            s = str(row.get("text_excerpt") or "").strip()
            if s:
                texts.append(s)

    dedup: list[str] = []
    seen: set[str] = set()
    for t in texts:
        compact = " ".join(t.split())
        if len(compact) < 20:
            continue
        if compact in seen:
            continue
        seen.add(compact)
        dedup.append(compact)
        if len(dedup) >= max_examples:
            break
    return dedup


def _ai_similarity_stage(
    page_text: str,
    family_meta: dict[str, Any] | None,
    ai_runtime: dict[str, Any] | None,
    ai_batch_size: int,
    ai_max_examples: int,
) -> dict[str, Any]:
    if not ai_runtime:
        return {
            "status": "disabled",
            "ai_similarity_pct": None,
            "model": None,
            "device": None,
            "candidate_count": 0,
        }

    query = " ".join(str(page_text or "").split())
    if len(query) < 20:
        return {
            "status": "unavailable",
            "ai_similarity_pct": None,
            "model": ai_runtime.get("model_name"),
            "device": ai_runtime.get("device"),
            "candidate_count": 0,
            "reason": "page_text_empty",
        }

    refs = _family_reference_texts(family_meta, ai_max_examples)
    if not refs:
        return {
            "status": "unavailable",
            "ai_similarity_pct": None,
            "model": ai_runtime.get("model_name"),
            "device": ai_runtime.get("device"),
            "candidate_count": 0,
            "reason": "family_reference_texts_missing",
        }

    model = ai_runtime.get("model")
    pairs = [(query, r) for r in refs]
    try:
        raw_scores = model.predict(
            pairs,
            batch_size=max(1, int(ai_batch_size)),
            show_progress_bar=False,
        )
    except Exception as exc:
        return {
            "status": "error",
            "ai_similarity_pct": None,
            "model": ai_runtime.get("model_name"),
            "device": ai_runtime.get("device"),
            "candidate_count": len(refs),
            "error": str(exc),
        }

    if isinstance(raw_scores, np.ndarray):
        raw_vals = [float(x) for x in raw_scores.reshape(-1).tolist()]
    elif isinstance(raw_scores, (list, tuple)):
        raw_vals = [float(x) for x in raw_scores]
    else:
        raw_vals = [float(raw_scores)]

    norm_vals: list[float] = []
    for v in raw_vals:
        nv = _normalize_ai_raw_score(v)
        if nv is not None:
            norm_vals.append(float(nv))

    if not norm_vals:
        return {
            "status": "unavailable",
            "ai_similarity_pct": None,
            "model": ai_runtime.get("model_name"),
            "device": ai_runtime.get("device"),
            "candidate_count": len(refs),
            "reason": "invalid_model_scores",
        }

    best_idx = int(np.argmax(np.array(norm_vals, dtype=float)))
    best_score = float(norm_vals[best_idx])
    mean_score = float(sum(norm_vals) / max(1, len(norm_vals)))
    return {
        "status": "ok",
        "ai_similarity_pct": float(round(best_score * 100.0, 2)),
        "model": ai_runtime.get("model_name"),
        "device": ai_runtime.get("device"),
        "candidate_count": len(refs),
        "best_candidate_index": best_idx,
        "best_candidate_excerpt": refs[best_idx][:220],
        "best_score": float(round(best_score, 6)),
        "mean_score": float(round(mean_score, 6)),
    }


def _score_family(
    z: np.ndarray,
    page_idx: int,
    page_ref: dict[str, Any],
    page_tokens: list[str] | None = None,
    page_text: str = "",
    ai_runtime: dict[str, Any] | None = None,
    ai_batch_size: int = AI_BATCH_SIZE_DEFAULT,
    ai_max_examples: int = AI_MAX_EXAMPLES_DEFAULT,
) -> dict[str, Any]:
    cents = np.array(page_ref.get("centroids_norm", []), dtype=float)
    if cents.size == 0:
        return _empty_result(page_idx, confidence="uncataloged_page", status="uncataloged_page")

    family_comparison: list[dict[str, Any]] = []
    page_weights = _tokens_to_weights(page_tokens or [], TOKEN_PAGE_MAX_TERMS)
    for cid in range(cents.shape[0]):
        centroid = cents[cid]
        delta = np.abs(centroid - z)
        distance = float(np.linalg.norm(delta))

        feature_similarity_pct: dict[str, float] = {}
        feature_delta_z: dict[str, float] = {}
        for fname, dval in zip(FEATURE_FIELDS, delta.tolist()):
            d = float(dval)
            feature_delta_z[fname] = d
            feature_similarity_pct[fname] = round(_feature_similarity_pct(d), 2)

        similarity_pct = float(
            round(
                sum(float(v) for v in feature_similarity_pct.values()) / max(1, len(feature_similarity_pct)),
                2,
            )
        )
        family_meta = _family_meta_from_page(page_ref, cid)
        confidence, family_status, thresholds, rule = _resolve_family_confidence(distance, family_meta)
        family_weights = _family_token_weights(family_meta)
        token_stage = _token_similarity_stage(page_weights, family_weights)
        ai_stage = _ai_similarity_stage(
            page_text,
            family_meta,
            ai_runtime=ai_runtime,
            ai_batch_size=ai_batch_size,
            ai_max_examples=ai_max_examples,
        )
        family_comparison.append(
            {
                "family_id": int(cid),
                "distance": distance,
                "similarity_pct": similarity_pct,
                "token_similarity_pct": token_stage["token_similarity_pct"],
                "token_stage": token_stage,
                "ai_similarity_pct": ai_stage.get("ai_similarity_pct"),
                "ai_stage": ai_stage,
                "confidence": confidence,
                "family_status": family_status,
                "thresholds": thresholds,
                "rule_applied": rule,
                "feature_similarity_pct": feature_similarity_pct,
                "feature_delta_z": feature_delta_z,
            }
        )

    ai_available = any(isinstance(row.get("ai_similarity_pct"), (int, float)) for row in family_comparison)
    if ai_runtime and ai_available:
        family_comparison.sort(
            key=lambda x: (
                -float(x.get("ai_similarity_pct") if isinstance(x.get("ai_similarity_pct"), (int, float)) else -1.0),
                float(x.get("distance", float("inf"))),
            )
        )
        selection_strategy = "ai_gpu_rerank_then_distance"
    else:
        family_comparison.sort(key=lambda x: float(x.get("distance", float("inf"))))
        selection_strategy = "distance_only"

    best_cmp = family_comparison[0]
    cid = int(best_cmp["family_id"])
    dists = [float(item["distance"]) for item in sorted(family_comparison, key=lambda x: int(x["family_id"]))]

    return {
        "page_index": page_idx,
        "catalog_page_index": CATALOG_PAGE_INDEX,
        "mode": "family_catalog",
        "family_id": cid,
        "cluster": cid,
        "family": cid,
        "distance": float(best_cmp["distance"]),
        "scores": [float(d) for d in dists],
        "confidence": str(best_cmp["confidence"]),
        "family_status": str(best_cmp["family_status"]),
        "thresholds": best_cmp["thresholds"],
        "rule_applied": str(best_cmp["rule_applied"]),
        "selection_strategy": selection_strategy,
        "family_similarity_pct": float(best_cmp["similarity_pct"]),
        "token_similarity_pct": best_cmp.get("token_similarity_pct"),
        "token_stage": dict(best_cmp.get("token_stage") or {}),
        "ai_similarity_pct": best_cmp.get("ai_similarity_pct"),
        "ai_stage": dict(best_cmp.get("ai_stage") or {}),
        "feature_similarity_pct": dict(best_cmp["feature_similarity_pct"]),
        "feature_delta_z": dict(best_cmp["feature_delta_z"]),
        "family_comparison": family_comparison,
    }


def score_page_against_catalog(
    feat: list[float],
    ref: dict,
    page_idx: int,
    page_tokens: list[str] | None = None,
    page_text: str = "",
    ai_runtime: dict[str, Any] | None = None,
    ai_batch_size: int = AI_BATCH_SIZE_DEFAULT,
    ai_max_examples: int = AI_MAX_EXAMPLES_DEFAULT,
) -> dict[str, Any]:
    # Política oficial: todo match usa o catálogo da página 0.
    page_ref = ref.get("pages", {}).get(str(CATALOG_PAGE_INDEX))
    if not page_ref:
        return _empty_result(
            page_idx,
            confidence="uncataloged_page",
            status="catalog_page_not_found",
        )

    ref_feature_fields = ref.get("feature_fields")
    if isinstance(ref_feature_fields, list):
        ref_feature_fields = [str(x) for x in ref_feature_fields]
    if ref_feature_fields and ref_feature_fields != FEATURE_FIELDS:
        raise ValueError(
            "catalogo incompatível com features atuais; "
            "refaça catalog_collect+catalog_register para gerar catálogo novo"
        )

    mean = np.array(page_ref.get("mean", []), dtype=float)
    std = np.array(
        [s if abs(float(s)) >= 1e-9 else 1.0 for s in page_ref.get("std", [])],
        dtype=float,
    )
    if len(feat) != len(mean):
        raise ValueError(f"feature length invalido: feat={len(feat)} ref={len(mean)}")

    z = (np.array(feat, dtype=float) - mean) / std

    mode = str(page_ref.get("classification_mode") or "family_catalog").strip()
    if mode != "family_catalog":
        raise ValueError(
            "catalogo invalido para classificação: esperado classification_mode=family_catalog"
        )
    return _score_family(
        z,
        page_idx,
        page_ref,
        page_tokens=page_tokens,
        page_text=page_text,
        ai_runtime=ai_runtime,
        ai_batch_size=ai_batch_size,
        ai_max_examples=ai_max_examples,
    )


def score_pdf_against_catalog(
    pdf: Path,
    ref: dict[str, Any],
    *,
    ai_model: str = AI_MODEL_DEFAULT,
    ai_batch_size: int = AI_BATCH_SIZE_DEFAULT,
    ai_max_examples: int = AI_MAX_EXAMPLES_DEFAULT,
) -> dict[str, Any]:
    ai_runtime: dict[str, Any] | None = None
    try:
        ai_runtime = _require_ai_gpu_runtime(str(ai_model))
    except Exception as exc:
        return {
            "pdf": str(pdf),
            "catalog_page_index": CATALOG_PAGE_INDEX,
            "status": "ai_runtime_error",
            "error": str(exc),
            "pages_total": 0,
            "pages_scanned": 0,
            "page_feature_errors_count": 0,
            "detected": False,
            "best_page": None,
            "pages": [],
            "ai_runtime": {
                "enabled": True,
                "model": str(ai_model),
                "device": "cuda",
                "cpu_fallback": False,
            },
        }

    try:
        doc = fitz.open(pdf)
    except Exception as exc:
        return {
            "pdf": str(pdf),
            "catalog_page_index": CATALOG_PAGE_INDEX,
            "status": "invalid_pdf",
            "error": str(exc),
            "pages_total": 0,
            "pages_scanned": 0,
            "page_feature_errors_count": 0,
            "detected": False,
            "best_page": None,
            "pages": [],
            "ai_runtime": {
                "enabled": True,
                "model": str(ai_model),
                "device": "cuda",
                "cpu_fallback": False,
            },
        }

    rows: list[dict[str, Any]] = []
    page_feature_errors = 0
    try:
        pages_total = int(doc.page_count)
        for source_page_idx in range(max(0, pages_total)):
            try:
                feat, _ = extract_page_features_from_doc(doc, source_page_idx)
            except Exception:
                page_feature_errors += 1
                continue
            if not feat:
                continue
            tokens = extract_page_tokens_from_doc(doc, source_page_idx)
            page_text = extract_page_text_from_doc(doc, source_page_idx)
            try:
                res = score_page_against_catalog(
                    feat,
                    ref,
                    source_page_idx,
                    page_tokens=tokens,
                    page_text=page_text,
                    ai_runtime=ai_runtime,
                    ai_batch_size=ai_batch_size,
                    ai_max_examples=ai_max_examples,
                )
            except Exception as exc:
                return {
                    "pdf": str(pdf),
                    "catalog_page_index": CATALOG_PAGE_INDEX,
                    "status": "classification_error",
                    "error": str(exc),
                    "pages_total": pages_total,
                    "pages_scanned": 0,
                    "page_feature_errors_count": int(page_feature_errors),
                    "detected": False,
                    "best_page": None,
                    "pages": [],
                    "ai_runtime": {
                        "enabled": True,
                        "model": str(ai_model),
                        "device": "cuda",
                        "cpu_fallback": False,
                    },
                }
            res["file"] = pdf.name
            res["features"] = {k: v for k, v in zip(FEATURE_FIELDS, feat)}
            rows.append(res)
    finally:
        try:
            doc.close()
        except Exception:
            pass

    if not rows:
        return {
            "pdf": str(pdf),
            "catalog_page_index": CATALOG_PAGE_INDEX,
            "status": "no_usable_pages",
            "pages_total": pages_total,
            "pages_scanned": 0,
            "page_feature_errors_count": int(page_feature_errors),
            "detected": False,
            "best_page": None,
            "pages": [],
            "ai_runtime": {
                "enabled": True,
                "model": str(ai_model),
                "device": "cuda",
                "cpu_fallback": False,
            },
        }

    def _score_row(r: dict[str, Any]) -> tuple[int, float, int, float]:
        ai = r.get("ai_similarity_pct")
        ai_rank = 1 if isinstance(ai, (int, float)) else 0
        ai_val = float(ai) if isinstance(ai, (int, float)) else -1.0
        conf_rank = _confidence_rank(str(r.get("confidence") or ""))
        distance = r.get("distance")
        if isinstance(distance, (int, float)):
            dist_val = float(distance)
        else:
            dist_val = float("inf")
        return ai_rank, ai_val, conf_rank, -dist_val

    best = max(rows, key=_score_row)
    hit_rows = [r for r in rows if _is_positive_confidence(str(r.get("confidence") or ""))]
    entry_rows = _entry_rows_from_hits(hit_rows)
    detected = bool(hit_rows)

    # Para documento multidoc: prioriza página de entrada de bloco de hit.
    if entry_rows:
        best = max(entry_rows, key=_score_row)

    return {
        "pdf": str(pdf),
        "catalog_page_index": CATALOG_PAGE_INDEX,
        "status": "detected" if detected else "not_detected",
        "pages_total": pages_total,
        "pages_scanned": int(len(rows)),
        "page_feature_errors_count": int(page_feature_errors),
        "detected": bool(detected),
        "hit_pages_count": int(len(hit_rows)),
        "hit_page_indices": [int(r.get("page_index")) for r in hit_rows if isinstance(r.get("page_index"), int)],
        "entry_pages_count": int(len(entry_rows)),
        "entry_page_indices": [int(r.get("page_index")) for r in entry_rows if isinstance(r.get("page_index"), int)],
        "best_page": best,
        "pages": rows,
        "ai_runtime": {
            "enabled": True,
            "model": str(ai_model),
            "device": "cuda",
            "cpu_fallback": False,
            "batch_size": int(ai_batch_size),
            "max_examples_per_family": int(ai_max_examples),
        },
    }


def run(
    pdf: Path,
    tag: str,
    registry_path: Path,
    *,
    ai_model: str,
    ai_batch_size: int,
    ai_max_examples: int,
) -> int:
    norm_tag = normalize_tag(tag)
    ref_path = resolve_catalog_by_tag(registry_path, norm_tag)
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    out = score_pdf_against_catalog(
        pdf,
        ref,
        ai_model=str(ai_model),
        ai_batch_size=int(ai_batch_size),
        ai_max_examples=int(ai_max_examples),
    )
    out["requested_tag"] = norm_tag
    out["catalog_path"] = str(ref_path)
    out["catalog_tag"] = ref.get("catalog_tag")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Similarity Score: pontua páginas do PDF contra o catálogo da capa com IA lexical em GPU obrigatória"
    )
    ap.add_argument("--pdf", required=True, help="PDF a pontuar")
    ap.add_argument("--tag", required=True, help="TAG já catalogada (ex.: despacho)")
    ap.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY_PATH),
        help="Registry oficial de tags catalogadas",
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
    return run(
        Path(args.pdf),
        str(args.tag),
        Path(args.registry),
        ai_model=str(args.ai_model),
        ai_batch_size=int(args.ai_batch_size),
        ai_max_examples=int(args.ai_max_examples),
    )


if __name__ == "__main__":
    raise SystemExit(main())
