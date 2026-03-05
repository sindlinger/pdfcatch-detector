from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CATALOG_ROOT = PROJECT_ROOT / "io" / "catalogacao_classica" / "catalogs"
CATALOG_INDEX_PATH = CATALOG_ROOT / "catalog_index.json"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_tag(tag: str) -> str:
    s = str(tag or "").strip().lower()
    s = re.sub(r"[^0-9a-zA-Z_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise ValueError("tag vazia/inválida")
    return s


def catalog_dir_for_tag(tag: str) -> Path:
    return CATALOG_ROOT / normalize_tag(tag)


def _empty_index() -> dict[str, Any]:
    return {
        "schema": "catalog_index.v1",
        "updated_at_utc": now_utc_iso(),
        "tags": {},
    }


def load_index(path: Path = CATALOG_INDEX_PATH) -> dict[str, Any]:
    if not path.exists():
        return _empty_index()
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"catalog index inválido: {path}")
    obj.setdefault("schema", "catalog_index.v1")
    obj.setdefault("updated_at_utc", now_utc_iso())
    obj.setdefault("tags", {})
    if not isinstance(obj["tags"], dict):
        raise ValueError(f"campo tags inválido em: {path}")
    return obj


def save_index(data: dict[str, Any], path: Path = CATALOG_INDEX_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["schema"] = "catalog_index.v1"
    payload["updated_at_utc"] = now_utc_iso()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def get_tag_entry(tag: str, path: Path = CATALOG_INDEX_PATH) -> dict[str, Any] | None:
    idx = load_index(path)
    return idx.get("tags", {}).get(normalize_tag(tag))


def set_tag_entry(tag: str, entry: dict[str, Any], path: Path = CATALOG_INDEX_PATH) -> Path:
    idx = load_index(path)
    tags = idx.setdefault("tags", {})
    tags[normalize_tag(tag)] = entry
    return save_index(idx, path)
