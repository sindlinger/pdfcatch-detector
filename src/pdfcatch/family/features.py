from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import fitz

FEATURE_FIELDS: List[str] = [
    "chars",
    "token_count",
    "fill_percent",
    "spaces",
    "whitespace",
    "non_whitespace",
    "line_count",
    "empty_lines",
    "font_count",
    "width_mm",
    "height_mm",
]

_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÿ]+", re.UNICODE)


def tokenize_text(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _extract_page_features_from_page(page: fitz.Page) -> list[float]:
    text = page.get_text("text") or ""
    tokens = tokenize_text(text)
    chars = len(text)
    token_count = len(tokens)
    spaces = text.count(" ") if text else 0
    whitespace = sum(1 for ch in text if ch.isspace()) if text else 0
    non_whitespace = chars - whitespace
    fill_percent = float(non_whitespace) / float(chars) if chars else 0.0
    lines = text.splitlines()
    line_count = len(lines)
    empty_lines = sum(1 for l in lines if l.strip() == "")
    try:
        fonts = [
            b.get("font", "")
            for b in page.get_text("dict").get("blocks", [])
            if b.get("type") == 0
            for l in b.get("lines", [])
            for s in l.get("spans", [])
        ]
        font_count = len(set(fonts))
    except Exception:
        font_count = 0
    width_mm = float(page.rect.width) * 25.4 / 72.0
    height_mm = float(page.rect.height) * 25.4 / 72.0

    return [
        float(chars),
        float(token_count),
        float(fill_percent),
        float(spaces),
        float(whitespace),
        float(non_whitespace),
        float(line_count),
        float(empty_lines),
        float(font_count),
        float(width_mm),
        float(height_mm),
    ]


def extract_page_features_from_doc(doc: fitz.Document, page_index: int) -> Tuple[list[float], int]:
    try:
        if page_index >= doc.page_count:
            return [], page_index
        page = doc.load_page(page_index)
        feat = _extract_page_features_from_page(page)
        return feat, page_index
    except Exception:
        return [], page_index


def extract_page_tokens_from_doc(doc: fitz.Document, page_index: int) -> list[str]:
    try:
        if page_index >= doc.page_count:
            return []
        page = doc.load_page(page_index)
        text = page.get_text("text") or ""
        return tokenize_text(text)
    except Exception:
        return []


def extract_page_text_from_doc(doc: fitz.Document, page_index: int) -> str:
    try:
        if page_index >= doc.page_count:
            return ""
        page = doc.load_page(page_index)
        return page.get_text("text") or ""
    except Exception:
        return ""


def extract_page_features(pdf_path: Path, page_index: int) -> Tuple[list[float], int]:
    doc = fitz.open(pdf_path)
    try:
        return extract_page_features_from_doc(doc, page_index)
    finally:
        try:
            doc.close()
        except Exception:
            pass


def normalize_matrix(X: list[list[float]]) -> tuple[list[float], list[float], list[list[float]]]:
    import numpy as np

    arr = np.array(X, dtype=float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    # Evita explosão numérica quando um campo é praticamente constante (ex.: width/height).
    std_adj = np.where(np.abs(std) < 1e-9, 1.0, std)
    Xn = (arr - mean) / std_adj
    return mean.tolist(), std.tolist(), Xn.tolist()
