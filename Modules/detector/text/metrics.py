from __future__ import annotations

from dataclasses import dataclass

from pdfcatch.lib.textsim import (
    bow_cosine,
    jaccard,
    normalize_for_similarity,
    order_similarity,
    tokenize,
)


@dataclass(frozen=True)
class TextScores:
    cosine: float
    jaccard: float
    hybrid: float
    order: float


def score_text(a_text: str, b_text: str) -> TextScores:
    a_norm = normalize_for_similarity(a_text or "")
    b_norm = normalize_for_similarity(b_text or "")
    a_toks = tokenize(a_norm)
    b_toks = tokenize(b_norm)

    c = bow_cosine(a_toks, b_toks)
    j = jaccard(a_toks, b_toks)
    h = max(0.0, min(1.0, 0.75 * c + 0.25 * j))
    o = order_similarity(a_text or "", b_text or "")
    return TextScores(cosine=float(c), jaccard=float(j), hybrid=float(h), order=float(o))

