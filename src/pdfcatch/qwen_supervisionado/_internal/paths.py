from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QWEN_ROOT = PROJECT_ROOT / "io" / "qwen_isolated"
QWEN_INPUT_ROOT = QWEN_ROOT / "input"
QWEN_CATALOG_ROOT = QWEN_ROOT / "catalogs"
QWEN_OUTPUT_ROOT = QWEN_ROOT / "outputs"
QWEN_MODEL_ROOT = QWEN_ROOT / "models"


def _resolve(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def ensure_inside(path: Path | str, root: Path | str, scope: str) -> Path:
    resolved_path = _resolve(path)
    resolved_root = _resolve(root)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"{scope} fora da area isolada do Qwen: "
            f"path={resolved_path} root={resolved_root}"
        ) from exc
    return resolved_path


def ensure_input_dir(path: Path | str) -> Path:
    resolved = ensure_inside(path, QWEN_INPUT_ROOT, "diretorio de entrada")
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"diretorio de entrada invalido: {resolved}")
    return resolved


def ensure_catalog_dir(path: Path | str) -> Path:
    resolved = ensure_inside(path, QWEN_CATALOG_ROOT, "diretorio de catalogacao")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_output_file(path: Path | str) -> Path:
    resolved = ensure_inside(path, QWEN_OUTPUT_ROOT, "arquivo de saida")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_model_dir(path: Path | str) -> Path:
    resolved = ensure_inside(path, QWEN_MODEL_ROOT, "diretorio de modelo")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_model_file(path: Path | str) -> Path:
    resolved = ensure_inside(path, QWEN_MODEL_ROOT, "arquivo de modelo")
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"arquivo de modelo invalido: {resolved}")
    return resolved


def bootstrap_layout() -> None:
    QWEN_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    QWEN_CATALOG_ROOT.mkdir(parents=True, exist_ok=True)
    QWEN_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    QWEN_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
