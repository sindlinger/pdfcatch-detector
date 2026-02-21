"""
Byte-level window fingerprinting helpers.

This module exposes the public API used by detection code with naming aligned
to the active methods (bytes/tlsh). It currently delegates to the legacy
implementation in `prefilter.py` for backward compatibility.
"""

from .prefilter import (
    FileFingerprint,
    build_model_window_pdf,
    extract_window_pdf_file,
    extract_windows_and_fingerprint,
    fingerprint_file,
    pick_best_window_by_tlsh,
    tlsh_available,
    tlsh_distance,
    tlsh_similarity,
)

__all__ = [
    "FileFingerprint",
    "build_model_window_pdf",
    "extract_window_pdf_file",
    "extract_windows_and_fingerprint",
    "fingerprint_file",
    "pick_best_window_by_tlsh",
    "tlsh_available",
    "tlsh_distance",
    "tlsh_similarity",
]

