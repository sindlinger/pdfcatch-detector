from __future__ import annotations

import sys
from pathlib import Path

# Ensure top-level `Modules/` package is importable in local/editable runs.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if (_REPO_ROOT / "Modules").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

__all__ = ["__version__"]

__version__ = "0.1.0"
