from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_CATALOG_DIR = PROJECT_ROOT / "io" / "catalog_official"
OFFICIAL_TAGS_DIR = OFFICIAL_CATALOG_DIR / "tags"
OFFICIAL_AUDIT_DIR = OFFICIAL_CATALOG_DIR / "audit"
DEFAULT_REGISTRY_PATH = OFFICIAL_CATALOG_DIR / "catalog_registry.json"
DEFAULT_UPDATE_REQUEST_PATH = PROJECT_ROOT / "atualizar catálogo.json"
UPDATE_REQUEST_CANDIDATES = [
    DEFAULT_UPDATE_REQUEST_PATH,
    PROJECT_ROOT / "atualizar catalogo.json",
    PROJECT_ROOT / "atualizar catálogo",
    PROJECT_ROOT / "atualizar catalogo",
]
LEGACY_REGISTRY_CANDIDATES = [
    PROJECT_ROOT / "io" / "out" / "_tmp_catalog_registry.json",
    PROJECT_ROOT / "io" / "out" / "catalog_registry.json",
]
REGISTRY_SCHEMA = "catalog_registry.v1"
_TAG_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _extract_catalog_signature(entry: dict[str, Any]) -> str | None:
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


def normalize_tag(tag: str) -> str:
    value = str(tag or "").strip()
    if not value:
        raise ValueError("tag obrigatoria")
    if not _TAG_RE.match(value):
        raise ValueError("tag invalida: use somente letras, numeros, ponto, underscore e hifen")
    return value.lower()


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        _resolve(path).relative_to(_resolve(parent))
        return True
    except Exception:
        return False


def ensure_official_dirs() -> None:
    OFFICIAL_TAGS_DIR.mkdir(parents=True, exist_ok=True)
    OFFICIAL_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)


def ensure_official_registry_path(path: Path) -> Path:
    requested = _resolve(path)
    official = _resolve(DEFAULT_REGISTRY_PATH)
    if requested != official:
        raise ValueError(
            "registry fora do diretório oficial foi bloqueado. "
            f"registry_oficial={official} registry_informado={requested}"
        )
    ensure_official_dirs()
    return official


def official_catalog_path_for_tag(tag: str) -> Path:
    norm_tag = normalize_tag(tag)
    ensure_official_dirs()
    return OFFICIAL_TAGS_DIR / norm_tag / "catalog.json"


def official_tag_history_dir(tag: str) -> Path:
    norm_tag = normalize_tag(tag)
    ensure_official_dirs()
    return OFFICIAL_TAGS_DIR / norm_tag / "history"


def _write_audit_snapshot(name: str, payload: dict[str, Any]) -> None:
    ensure_official_dirs()
    out = OFFICIAL_AUDIT_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _empty_registry() -> dict[str, Any]:
    return {
        "schema": REGISTRY_SCHEMA,
        "updated_at_utc": now_iso_utc(),
        "tags": {},
    }


def _bootstrap_official_registry_from_legacy(registry_path: Path) -> None:
    if registry_path.exists():
        return

    migrated_tags: dict[str, Any] = {}
    migration_details: list[dict[str, Any]] = []
    for legacy in LEGACY_REGISTRY_CANDIDATES:
        legacy_path = _resolve(legacy)
        if not legacy_path.exists():
            continue
        try:
            raw = json.loads(legacy_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        tags = raw.get("tags") if isinstance(raw, dict) else None
        if not isinstance(tags, dict):
            continue

        for raw_tag, entry in tags.items():
            if not isinstance(entry, dict):
                continue
            try:
                norm_tag = normalize_tag(str(raw_tag))
            except Exception:
                continue

            src_catalog = entry.get("catalog_path")
            if not src_catalog:
                continue
            src_catalog_path = _resolve(Path(str(src_catalog)))
            if not src_catalog_path.exists():
                continue

            dst_catalog = _resolve(official_catalog_path_for_tag(norm_tag))
            dst_catalog.parent.mkdir(parents=True, exist_ok=True)
            if src_catalog_path != dst_catalog:
                shutil.copy2(src_catalog_path, dst_catalog)

            migrated_tags[norm_tag] = {
                "tag": norm_tag,
                "catalog_path": str(dst_catalog),
                "catalog_signature": _extract_catalog_signature(entry),
                "samples": entry.get("samples"),
                "created_at_utc": entry.get("created_at_utc") or now_iso_utc(),
                "updated_at_utc": now_iso_utc(),
                "update_count": int(entry.get("update_count") or 0),
                "status": "migrated_from_legacy",
                "legacy_registry_path": str(legacy_path),
                "legacy_catalog_path": str(src_catalog_path),
            }
            migration_details.append(
                {
                    "tag": norm_tag,
                    "legacy_registry_path": str(legacy_path),
                    "legacy_catalog_path": str(src_catalog_path),
                    "official_catalog_path": str(dst_catalog),
                }
            )

    if migrated_tags:
        data = _empty_registry()
        data["tags"] = migrated_tags
        save_registry(registry_path, data)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _write_audit_snapshot(
            f"legacy_migration__{stamp}.json",
            {
                "schema": "catalog_registry.legacy_migration.v1",
                "migrated_at_utc": now_iso_utc(),
                "registry_path": str(_resolve(registry_path)),
                "migrated_count": len(migration_details),
                "items": migration_details,
            },
        )


def load_registry(path: Path) -> dict[str, Any]:
    path = ensure_official_registry_path(path)
    _bootstrap_official_registry_from_legacy(path)
    if not path.exists():
        return _empty_registry()
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"registry invalido: {path}")
    if "tags" not in obj or not isinstance(obj.get("tags"), dict):
        raise ValueError(f"registry invalido: campo 'tags' ausente ou invalido em {path}")
    if not obj.get("schema"):
        obj["schema"] = REGISTRY_SCHEMA
    return obj


def save_registry(path: Path, data: dict[str, Any]) -> None:
    path = ensure_official_registry_path(path)
    data = dict(data)
    data["schema"] = REGISTRY_SCHEMA
    data["updated_at_utc"] = now_iso_utc()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def register_catalog_tag(
    *,
    registry_path: Path,
    tag: str,
    catalog_path: Path,
    catalog_signature: str | None,
    samples: int | None,
) -> dict[str, Any]:
    registry_path = ensure_official_registry_path(registry_path)
    norm_tag = normalize_tag(tag)
    expected_catalog = _resolve(official_catalog_path_for_tag(norm_tag))
    resolved_catalog = _resolve(catalog_path)
    if resolved_catalog != expected_catalog:
        raise ValueError(
            "catalog_path fora do diretório oficial foi bloqueado. "
            f"catalog_oficial={expected_catalog} catalog_informado={resolved_catalog}"
        )

    reg = load_registry(registry_path)
    tags = reg["tags"]
    if norm_tag in tags:
        entry = tags[norm_tag]
        raise ValueError(
            "tag ja catalogada; nova criacao bloqueada para evitar sobrescrita. "
            f"tag={norm_tag} catalog_path={entry.get('catalog_path')}"
        )

    tags[norm_tag] = {
        "tag": norm_tag,
        "catalog_path": str(resolved_catalog),
        "catalog_signature": str(catalog_signature) if catalog_signature else None,
        "samples": int(samples) if isinstance(samples, int) else None,
        "created_at_utc": now_iso_utc(),
        "updated_at_utc": now_iso_utc(),
        "update_count": 0,
    }
    save_registry(registry_path, reg)
    return tags[norm_tag]


def update_catalog_tag(
    *,
    registry_path: Path,
    tag: str,
    catalog_path: Path,
    catalog_signature: str | None,
    samples: int | None,
    audit_event_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry_path = ensure_official_registry_path(registry_path)
    norm_tag = normalize_tag(tag)
    expected_catalog = _resolve(official_catalog_path_for_tag(norm_tag))
    resolved_catalog = _resolve(catalog_path)
    if resolved_catalog != expected_catalog:
        raise ValueError(
            "catalog_path fora do diretório oficial foi bloqueado. "
            f"catalog_oficial={expected_catalog} catalog_informado={resolved_catalog}"
        )

    reg = load_registry(registry_path)
    tags = reg["tags"]
    if norm_tag not in tags:
        raise ValueError(f"tag nao catalogada: {norm_tag}")
    entry = dict(tags[norm_tag] or {})
    entry["tag"] = norm_tag
    entry["catalog_path"] = str(resolved_catalog)
    entry["catalog_signature"] = str(catalog_signature) if catalog_signature else None
    for key in list(entry.keys()):
        if str(key).endswith("_signature") and str(key) != "catalog_signature":
            entry.pop(key, None)
    entry["samples"] = int(samples) if isinstance(samples, int) else None
    entry["updated_at_utc"] = now_iso_utc()
    entry["update_count"] = int(entry.get("update_count") or 0) + 1
    if audit_event_path is not None:
        entry["last_audit_event_path"] = str(_resolve(audit_event_path))
    if metadata:
        for k, v in metadata.items():
            entry[k] = v
    tags[norm_tag] = entry
    save_registry(registry_path, reg)
    return entry


def resolve_update_request_path(path: Path | None = None) -> Path:
    if path is not None:
        return _resolve(path)
    for candidate in UPDATE_REQUEST_CANDIDATES:
        if candidate.exists():
            return _resolve(candidate)
    return _resolve(DEFAULT_UPDATE_REQUEST_PATH)


def load_update_request(path: Path | None = None) -> tuple[Path, dict[str, Any] | None]:
    req_path = resolve_update_request_path(path)
    if not req_path.exists():
        return req_path, None
    try:
        raw = json.loads(req_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"arquivo de atualização inválido ({req_path}): {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"arquivo de atualização inválido ({req_path}): esperado objeto JSON")
    return req_path, raw


def consume_update_request(path: Path, archive_path: Path) -> Path | None:
    src = _resolve(path)
    if not src.exists():
        return None
    dst = _resolve(archive_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return dst


def resolve_catalog_by_tag(registry_path: Path, tag: str) -> Path:
    registry_path = ensure_official_registry_path(registry_path)
    norm_tag = normalize_tag(tag)
    reg = load_registry(registry_path)
    tags = reg.get("tags") or {}
    entry = tags.get(norm_tag)
    if not isinstance(entry, dict):
        raise ValueError(f"tag nao catalogada: {norm_tag}")
    cpath = entry.get("catalog_path")
    if not cpath:
        raise ValueError(f"registry inconsistente para tag {norm_tag}: catalog_path ausente")
    out = Path(str(cpath))
    if not _is_within(out, OFFICIAL_CATALOG_DIR):
        raise ValueError(
            "catalogo fora do diretório oficial foi bloqueado. "
            f"tag={norm_tag} catalog_path={_resolve(out)} official_root={_resolve(OFFICIAL_CATALOG_DIR)}"
        )
    if not out.exists():
        raise ValueError(f"catalogo da tag {norm_tag} nao encontrado em: {out}")
    return out
