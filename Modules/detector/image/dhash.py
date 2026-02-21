from __future__ import annotations

import fitz  # PyMuPDF

from pdfcatch.lib.imghash import DHash, dhash_from_page, dhash_similarity


def page_dhash(doc: fitz.Document, *, page_index: int, hash_size: int = 8, dpi: int = 72) -> DHash | None:
    return dhash_from_page(doc.load_page(int(page_index)), hash_size=int(hash_size), dpi=int(dpi))


def dhash_score(a: DHash, b: DHash) -> float:
    score, _dist = dhash_similarity(a, b)
    return float(score)

