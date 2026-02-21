from __future__ import annotations

"""
Central entry module for the similarity engine (library-only).

No CLI is provided in this repo by design.
"""

from pdfcatch.core.engine import score_pdf_against_templates
from pdfcatch.core.model import EngineOptions, TemplateRef

__all__ = ["EngineOptions", "TemplateRef", "score_pdf_against_templates"]

