from __future__ import annotations

import fitz
from pathlib import Path
from typing import List, Tuple

# Conjunto fixo de 12 features por página (sem qr/signature)
FEATURE_FIELDS: List[str] = [
    "page_bytes",
    "chars",
    "fill_percent",
    "spaces",
    "whitespace",
    "non_whitespace",
    "line_count",
    "empty_lines",
    "font_count",
    "width_mm",
    "height_mm",
    "images",
]


def extract_page_features(pdf_path: Path, page_index: int) -> Tuple[list[float], int]:
    doc = fitz.open(pdf_path)
    try:
        if page_index >= doc.page_count:
            return [], page_index
        page = doc.load_page(page_index)
        # bytes da página regravada
        try:
            one = fitz.open()
            one.insert_pdf(doc, from_page=page_index, to_page=page_index)
            page_bytes = len(one.tobytes(garbage=4, clean=True, deflate=True, no_new_id=True, preserve_metadata=0))
        finally:
            try:
                one.close()
            except Exception:
                pass
        text = page.get_text("text") or ""
        chars = len(text)
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
        images = len(page.get_images(full=True) or [])
        width_mm = float(page.rect.width) * 25.4 / 72.0
        height_mm = float(page.rect.height) * 25.4 / 72.0

        feat = [
            float(page_bytes),
            float(chars),
            float(fill_percent),
            float(spaces),
            float(whitespace),
            float(non_whitespace),
            float(line_count),
            float(empty_lines),
            float(font_count),
            float(width_mm),
            float(height_mm),
            float(images),
        ]
        return feat, page_index
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
    std_adj = np.where(std == 0, 1.0, std)
    Xn = (arr - mean) / std_adj
    return mean.tolist(), std.tolist(), Xn.tolist()
