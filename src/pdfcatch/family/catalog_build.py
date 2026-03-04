from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from pdfcatch.family.catalog_registry import (
    DEFAULT_REGISTRY_PATH,
    DEFAULT_UPDATE_REQUEST_PATH,
    consume_update_request,
    ensure_official_registry_path,
    load_registry,
    load_update_request,
    normalize_tag,
    now_iso_utc,
    official_catalog_path_for_tag,
    official_tag_history_dir,
    register_catalog_tag,
    save_registry,
    update_catalog_tag,
)
from pdfcatch.family.features import (
    FEATURE_FIELDS,
    extract_page_text_from_doc,
    extract_page_tokens_from_doc,
    normalize_matrix,
)

DOCUMENT_REGISTRY_CONTRACT = {
    "phase_1_name": "family_catalog",
    "phase_1_objective": (
        "registrar referencias de familias de documentos conhecidos para responder "
        "'a qual familia conhecida este documento pertence?'"
    ),
    "phase_1_primary_output": "family_registry",
    "phase_1_non_objectives": [
        "nao inferir em diretorio misto",
        "nao procurar documentos desconhecidos fora do catalogo",
        "nao substituir catalogo oficial sem decisao explicita",
    ],
    "phase_2_name": "inference_search",
    "phase_2_objective": "buscar similaridade em diretorio misto/quarentena sem forcar familia conhecida",
}

JSON_FIELD_NAMING = {
    "family_identifier_field": "family_id",
    "family_registry_field": "family_registry",
    "family_distance_mean_field": "family_dist_mean",
    "family_distance_max_field": "family_dist_max",
    "family_reference_examples_field": "family_reference_examples",
    "similarity_output_family_field": "family_id",
}

CLASSIFICATION_CONTRACT = {
    "version": "family_distance_thresholds.v3",
    "same_page_required": True,
    "catalog_page_fixed": 0,
    "distance_metric": "euclidean_on_normalized_features",
    "mode_routing": "catalog_only",
    "inference_search_scope": "separate_phase_outside_catalog",
    "token_similarity_stage": {
        "enabled": True,
        "selection_impact": "advisory_separate_stage",
        "family_token_model": "tf_top_terms_weighted",
        "similarity_metrics": ["weighted_jaccard", "cosine"],
        "primary_output_field": "token_similarity_pct",
    },
    "ai_lexical_stage": {
        "enabled": True,
        "gpu_required": True,
        "selection_impact": "ai_gpu_rerank",
        "primary_output_field": "ai_similarity_pct",
    },
    "modes": {
        "family_catalog": {
            "selection": "nearest_centroid",
            "confidence_rules": {
                "high": "distance <= dist_mean",
                "medium": "dist_mean < distance <= dist_max",
                "low_out_of_family": "distance > dist_max",
            },
        },
    },
}

TOKEN_PROFILE_MAX_TERMS = 256
TOKEN_REFERENCE_TEXTS_MAX = 6
TOKEN_REFERENCE_EXCERPT_CHARS = 1800

def _warn(msg: str) -> None:
    print(f"catalog_register: aviso: {msg}")


def _resolve_existing_signature(entry: dict[str, Any]) -> str | None:
    primary = entry.get("catalog_signature")
    if isinstance(primary, str) and primary.strip():
        return primary.strip()
    for key, value in entry.items():
        if key == "catalog_signature":
            continue
        if not str(key).endswith("_signature"):
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _now_compact_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_json(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_pdf_path_from_row(row: pd.Series) -> Path | None:
    file_name = str(row.get("file") or "").strip()
    source_dir = str(row.get("source_dir") or "").strip()
    candidates: list[Path] = []

    if file_name:
        file_path = Path(file_name)
        if file_path.is_absolute():
            candidates.append(file_path)
        if source_dir:
            candidates.append(Path(source_dir) / file_name)
        candidates.append(Path.cwd() / file_name)

    for cand in candidates:
        try:
            resolved = cand.expanduser().resolve()
        except Exception:
            continue
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _normalize_excerpt(text: str, max_chars: int = TOKEN_REFERENCE_EXCERPT_CHARS) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    compact = " ".join(raw.split())
    return compact[:max_chars]


def _collect_row_lexical(sub: pd.DataFrame) -> tuple[list[list[str]], list[str]]:
    rows_tokens: list[list[str]] = []
    rows_texts: list[str] = []
    cache: dict[str, tuple[list[str], str]] = {}
    missing_paths = 0
    read_errors = 0

    for _, row in sub.iterrows():
        pdf_path = _resolve_pdf_path_from_row(row)
        if pdf_path is None:
            missing_paths += 1
            rows_tokens.append([])
            rows_texts.append("")
            continue

        key = str(pdf_path)
        if key in cache:
            toks, txt = cache[key]
            rows_tokens.append(toks)
            rows_texts.append(txt)
            continue

        tokens: list[str] = []
        text = ""
        try:
            doc = fitz.open(pdf_path)
            try:
                tokens = extract_page_tokens_from_doc(doc, 0)
                text = extract_page_text_from_doc(doc, 0)
            finally:
                try:
                    doc.close()
                except Exception:
                    pass
        except Exception:
            read_errors += 1
            tokens = []
            text = ""

        cache[key] = (tokens, text)
        rows_tokens.append(tokens)
        rows_texts.append(text)

    if missing_paths:
        _warn(
            "nao foi possível resolver o caminho de "
            f"{missing_paths} amostras para montar perfil lexical; "
            "token_similarity e ai_similarity serão parciais para essas entradas"
        )
    if read_errors:
        _warn(
            f"falha ao ler {read_errors} PDFs para perfil lexical; "
            "token_similarity e ai_similarity serão parciais para essas entradas"
        )

    return rows_tokens, rows_texts


def _build_family_token_model(family_tokens: list[list[str]], family_texts: list[str]) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    for toks in family_tokens:
        counter.update(toks)
    reference_texts = [_normalize_excerpt(t) for t in family_texts]
    reference_texts = [t for t in reference_texts if t][:TOKEN_REFERENCE_TEXTS_MAX]

    total_tokens = int(sum(counter.values()))
    unique_tokens = int(len(counter))
    if total_tokens <= 0:
        return {
            "method": "tf_top_terms_weighted",
            "total_tokens": 0,
            "unique_tokens": 0,
            "terms_kept": 0,
            "terms": [],
            "reference_texts": reference_texts,
        }

    terms = []
    kept_total = 0
    for token, count in counter.most_common(TOKEN_PROFILE_MAX_TERMS):
        c = int(count)
        kept_total += c
        terms.append(
            {
                "token": token,
                "count": c,
                "weight": float(c / total_tokens),
            }
        )

    return {
        "method": "tf_top_terms_weighted",
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "terms_kept": int(len(terms)),
        "terms_coverage_pct": float((kept_total / total_tokens) * 100.0),
        "terms": terms,
        "reference_texts": reference_texts,
    }


def _build_catalog_manifest(df: pd.DataFrame, csv_path: Path) -> dict[str, Any]:
    source_dirs: list[str] = []
    if "source_dir" in df.columns:
        source_dirs = sorted({str(v).strip() for v in df["source_dir"].tolist() if str(v).strip()})
    if not source_dirs:
        source_dirs = [f"csv:{csv_path.resolve()}"]

    files: list[str] = []
    if "file" in df.columns:
        files = sorted({str(v).strip() for v in df["file"].tolist() if str(v).strip()})

    page_counts: dict[str, int] = {}
    if "page_index" in df.columns:
        vc = df["page_index"].value_counts().to_dict()
        page_counts = {str(int(k)): int(v) for k, v in vc.items()}

    signature_payload = {
        "source_dirs": source_dirs,
        "files": files,
        "page_counts": page_counts,
        "rows_total": int(len(df)),
    }
    signature = _sha256_json(signature_payload)

    return {
        "source_dirs": source_dirs,
        "files_count": int(len(files)),
        "rows_total": int(len(df)),
        "page_counts": page_counts,
        "signature": signature,
        "signature_basis": signature_payload,
    }


def _read_update_request_for_tag(
    *,
    requested_tag: str,
    update_request_path: Path | None,
) -> tuple[Path, dict[str, Any] | None]:
    req_path, req = load_update_request(update_request_path)
    if req is None:
        return req_path, None

    req_tag_raw = req.get("tag")
    try:
        req_tag = normalize_tag(str(req_tag_raw or ""))
    except Exception as exc:
        raise ValueError(
            "arquivo de atualização inválido: campo 'tag' ausente/inválido. "
            f"arquivo={req_path}"
        ) from exc
    if req_tag != requested_tag:
        raise ValueError(
            "arquivo de atualização inválido para esta execução: "
            f"tag_arquivo={req_tag} tag_execucao={requested_tag} arquivo={req_path}"
        )
    if not bool(req.get("autorizar_atualizacao")):
        raise ValueError(
            "arquivo de atualização existe, mas não autorizou atualização. "
            "use 'autorizar_atualizacao: true'. "
            f"arquivo={req_path}"
        )
    return req_path, req


def _build_family_catalog(
    *,
    sub: pd.DataFrame,
    X: np.ndarray,
    Xn_arr: np.ndarray,
    row_tokens: list[list[str]],
    row_texts: list[str],
    k: int,
    examples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(Xn_arr)
    cent = km.cluster_centers_.tolist()
    families: dict[str, Any] = {}
    family_registry: list[dict[str, Any]] = []

    for cid in range(k):
        idxs = np.where(labels == cid)[0]
        if len(idxs) == 0:
            continue
        members = []
        for i in idxs:
            feats = X[i].tolist()
            z = Xn_arr[i]
            dist = float(np.linalg.norm(z - km.cluster_centers_[cid]))
            text_excerpt = _normalize_excerpt(row_texts[i])
            members.append(
                {
                    "file": str(sub.iloc[i]["file"]) if "file" in sub.columns else "",
                    "page_index": int(sub.iloc[i]["page_index"]),
                    "features": feats,
                    "dist_norm": dist,
                    "text_excerpt": text_excerpt,
                }
            )
        members.sort(key=lambda m: m["dist_norm"])
        dvals = [m["dist_norm"] for m in members]
        token_model = _build_family_token_model(
            [row_tokens[i] for i in idxs],
            [row_texts[i] for i in idxs],
        )
        family_rec = {
            "family_id": int(cid),
            "family_centroid_norm": cent[cid],
            "family_member_count": int(len(members)),
            "family_dist_min": float(np.min(dvals)),
            "family_dist_mean": float(np.mean(dvals)),
            "family_dist_max": float(np.max(dvals)),
            "family_reference_examples": members[: min(len(members), examples)],
            "family_token_model": token_model,
        }
        family_registry.append(family_rec)
        # Compatibilidade com consumidores antigos.
        families[str(cid)] = {
            "family_id": int(cid),
            "centroid_norm": family_rec["family_centroid_norm"],
            "member_count": family_rec["family_member_count"],
            "dist_min": family_rec["family_dist_min"],
            "dist_mean": family_rec["family_dist_mean"],
            "dist_max": family_rec["family_dist_max"],
            "examples": family_rec["family_reference_examples"],
            "token_model": token_model,
        }

    family_registry.sort(key=lambda r: int(r["family_id"]))

    page_ref = {
        "classification_mode": "family_catalog",
        "catalog_objective": "document_family_registry",
        "k": int(k),
        "mean": [float(v) for v in np.mean(X, axis=0).tolist()],
        "std": [float(v) for v in np.std(X, axis=0).tolist()],
        "centroids_norm": cent,
        "family_registry": family_registry,
        "families": families,
    }
    summary = {
        "classification_mode": "family_catalog",
        "k": int(k),
        "samples": int(len(sub)),
        "families": int(len(families)),
    }
    return page_ref, summary


def run(
    csv_path: Path,
    k: int,
    examples: int,
    out: Path | None,
    tag: str,
    registry_path: Path,
    update_request_path: Path | None,
) -> int:
    registry_path = ensure_official_registry_path(registry_path)
    df = pd.read_csv(csv_path)
    if "page_index" not in df.columns:
        raise ValueError("csv precisa da coluna page_index")

    # Política oficial: catálogo de famílias baseado somente na página 0.
    page0_mask = df["page_index"] == 0
    dropped = int((~page0_mask).sum())
    if dropped > 0:
        _warn(
            "política de página única ativa: "
            f"ignorando {dropped} linhas com page_index != 0"
        )
    df = df[page0_mask].copy()
    if df.empty:
        raise ValueError("csv não possui linhas com page_index=0 para montar o catálogo")

    norm_tag = normalize_tag(tag)
    official_out = official_catalog_path_for_tag(norm_tag)
    if out is not None and out.resolve() != official_out.resolve():
        raise ValueError(
            "catálogo fora do diretório oficial foi bloqueado. "
            f"catalogo_oficial={official_out.resolve()} out_informado={out.resolve()}"
        )

    manifest = _build_catalog_manifest(df, csv_path)
    new_sig = str(manifest["signature"])
    reg = load_registry(registry_path)
    tags = reg.get("tags") or {}
    existing = tags.get(norm_tag)
    action = "create"
    req_path: Path | None = None
    req_payload: dict[str, Any] | None = None
    old_catalog_path: Path | None = None
    old_signature: str | None = None

    if existing:
        action = "update"
        req_path, req_payload = _read_update_request_for_tag(
            requested_tag=norm_tag,
            update_request_path=update_request_path,
        )
        if req_payload is None:
            raise ValueError(
                "tag ja catalogada; atualização bloqueada por padrão. "
                f"para atualizar esta TAG, forneça o arquivo '{DEFAULT_UPDATE_REQUEST_PATH.name}' "
                "com os campos obrigatórios: {\"tag\": \"...\", \"autorizar_atualizacao\": true}."
            )
        old_catalog = existing.get("catalog_path")
        if old_catalog:
            old_catalog_path = Path(str(old_catalog))
        old_signature = _resolve_existing_signature(existing)
    write_out = official_out

    ref: dict[str, Any] = {
        "schema": "bytes_family_kmeans.v2",
        "catalog_tag": norm_tag,
        "document_registry_contract": DOCUMENT_REGISTRY_CONTRACT,
        "json_field_naming": JSON_FIELD_NAMING,
        "feature_fields": FEATURE_FIELDS,
        "classification_contract": CLASSIFICATION_CONTRACT,
        "catalog_manifest": manifest,
        "output_policy": {
            "official_catalog_only": True,
            "requested_out": str(out.resolve()) if out is not None else str(official_out.resolve()),
            "written_out": str(write_out),
            "signature_existing": str(old_signature) if old_signature else None,
            "signature_new": new_sig,
            "signature_change_action": action,
            "update_request_file": str(req_path) if req_path is not None else None,
        },
        "pages": {},
        "summary": {},
    }

    page_idx = 0
    sub = df
    X = sub[FEATURE_FIELDS].to_numpy(dtype=float)
    row_tokens, row_texts = _collect_row_lexical(sub)
    mean, std, Xn = normalize_matrix(X.tolist())
    Xn_arr = np.array(Xn, dtype=float)
    if len(sub) < int(k):
        raise ValueError(
            f"page_index={page_idx} possui {len(sub)} amostras e k={k}; "
            "catálogo de famílias exige amostras >= k"
        )

    page_ref, summary = _build_family_catalog(
        sub=sub,
        X=X,
        Xn_arr=Xn_arr,
        row_tokens=row_tokens,
        row_texts=row_texts,
        k=int(k),
        examples=int(examples),
    )
    page_ref["mean"] = mean
    page_ref["std"] = std
    page_ref["mode_reason"] = "catalog_only"
    ref["pages"][str(page_idx)] = page_ref

    summary["mode_reason"] = "catalog_only"
    ref["summary"][str(page_idx)] = summary

    event_id = f"{_now_compact_utc()}__{action}__{new_sig[:12]}"
    event_dir = official_tag_history_dir(norm_tag) / event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    before_path = event_dir / "catalog_before.json"
    after_path = event_dir / "catalog_after.json"
    event_path = event_dir / "event.json"
    consumed_update_path: Path | None = None
    if old_catalog_path is not None and old_catalog_path.exists():
        shutil.copy2(old_catalog_path, before_path)

    write_out.parent.mkdir(parents=True, exist_ok=True)
    write_out.write_text(json.dumps(ref, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(write_out, after_path)

    if req_path is not None and req_path.exists():
        consumed_update_path = consume_update_request(req_path, event_dir / req_path.name)

    event_payload = {
        "schema": "catalog_event.v1",
        "event_id": event_id,
        "action": action,
        "executed_at_utc": now_iso_utc(),
        "tag": norm_tag,
        "registry_path": str(registry_path.resolve()),
        "official_catalog_path": str(write_out.resolve()),
        "catalog_before_path": str(before_path.resolve()) if before_path.exists() else None,
        "catalog_after_path": str(after_path.resolve()),
        "signature_before": old_signature,
        "signature_after": new_sig,
        "source_csv_path": str(csv_path.resolve()),
        "catalog_manifest": manifest,
        "runtime_context": {
            "cwd": str(Path.cwd().resolve()),
        },
        "update_request_source_path": str(req_path.resolve()) if req_path is not None else None,
        "update_request_consumed_path": str(consumed_update_path.resolve()) if consumed_update_path is not None else None,
        "update_request_payload": req_payload,
    }
    event_path.write_text(json.dumps(event_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if action == "create":
        register_catalog_tag(
            registry_path=registry_path,
            tag=norm_tag,
            catalog_path=write_out,
            catalog_signature=manifest.get("signature"),
            samples=int(len(sub)),
        )
        reg_after = load_registry(registry_path)
        tags_after = reg_after.get("tags") or {}
        entry_after = dict(tags_after.get(norm_tag) or {})
        entry_after["source_dirs"] = manifest.get("source_dirs")
        entry_after["files_count"] = manifest.get("files_count")
        entry_after["rows_total"] = manifest.get("rows_total")
        entry_after["last_action"] = "create"
        entry_after["last_audit_event_path"] = str(event_path.resolve())
        entry_after["history"] = [
            {
                "event_id": event_id,
                "event_path": str(event_path.resolve()),
                "executed_at_utc": event_payload["executed_at_utc"],
                "action": action,
                "signature_before": old_signature,
                "signature_after": new_sig,
            }
        ]
        tags_after[norm_tag] = entry_after
        save_registry(registry_path, reg_after)
    else:
        update_catalog_tag(
            registry_path=registry_path,
            tag=norm_tag,
            catalog_path=write_out,
            catalog_signature=manifest.get("signature"),
            samples=int(len(sub)),
            audit_event_path=event_path,
            metadata={
                "source_dirs": manifest.get("source_dirs"),
                "files_count": manifest.get("files_count"),
                "rows_total": manifest.get("rows_total"),
                "last_action": "update_by_update_request",
            },
        )
        # mantém histórico resumido no registry para consulta rápida
        reg_after = load_registry(registry_path)
        tags_after = reg_after.get("tags") or {}
        entry_after = dict(tags_after.get(norm_tag) or {})
        hist = entry_after.get("history")
        if not isinstance(hist, list):
            hist = []
        hist.append(
            {
                "event_id": event_id,
                "event_path": str(event_path.resolve()),
                "executed_at_utc": event_payload["executed_at_utc"],
                "action": action,
                "signature_before": old_signature,
                "signature_after": new_sig,
            }
        )
        entry_after["history"] = hist[-200:]
        entry_after["last_audit_event_path"] = str(event_path.resolve())
        tags_after[norm_tag] = entry_after
        save_registry(registry_path, reg_after)

    print(f"catalog_register: tag registrada '{norm_tag}' em {registry_path}")
    print(f"catalog_register: salvo {write_out} (assinatura={manifest['signature'][:12]})")
    print(f"catalog_register: auditoria salva em {event_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Catalog Register: cria catálogo de famílias usando somente page_index=0 (sem inferência)"
    )
    ap.add_argument("--csv", required=True, help="CSV gerado no catalog_collect")
    ap.add_argument("--k", type=int, default=3, help="k para o catálogo (página 0)")
    # Compatibilidade com comandos antigos.
    ap.add_argument("--k1", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--k2", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--examples", type=int, default=5, help="membros/exemplos salvos por grupo")
    ap.add_argument("--tag", required=True, help="TAG do documento catalogado (ex.: despacho)")
    ap.add_argument(
        "--out",
        default=None,
        help="Arquivo do catálogo (opcional). deve ser o caminho oficial da tag",
    )
    ap.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY_PATH),
        help="Registry oficial de tags catalogadas (fixo)",
    )
    ap.add_argument(
        "--update-request",
        default=str(DEFAULT_UPDATE_REQUEST_PATH),
        help="Arquivo de autorização para atualizar TAG existente (ex.: 'atualizar catálogo.json')",
    )
    args = ap.parse_args(argv)
    k_final = int(args.k)
    if args.k1 is not None:
        k_final = int(args.k1)
    if args.k2 is not None and int(args.k2) != int(k_final):
        ap.error("política de página única: --k2 não pode diferir de --k/--k1")
    return run(
        Path(args.csv),
        int(k_final),
        int(args.examples),
        Path(args.out) if args.out else None,
        str(args.tag),
        Path(args.registry),
        Path(args.update_request) if args.update_request else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
