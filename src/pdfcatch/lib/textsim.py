from __future__ import annotations

import difflib
import math
import re
from collections import Counter


_WS_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d")
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+", re.UNICODE)

_PT_STOPWORDS = {
    "a",
    "o",
    "os",
    "as",
    "de",
    "da",
    "das",
    "do",
    "dos",
    "e",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "por",
    "para",
    "com",
    "sem",
    "ao",
    "aos",
    "à",
    "às",
    "um",
    "uma",
    "uns",
    "umas",
}


def normalize_for_similarity(text: str) -> str:
    # Keep diacritics; only normalize whitespace and digits.
    text = text.lower()
    text = _DIGIT_RE.sub("#", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return tokenize_opts(text, remove_stopwords=True)


def tokenize_opts(text: str, *, remove_stopwords: bool) -> list[str]:
    toks = []
    for m in _TOKEN_RE.finditer(text):
        t = m.group(0)
        if len(t) <= 1:
            continue
        if remove_stopwords and t in _PT_STOPWORDS:
            continue
        toks.append(t)
    return toks


def bow_cosine(a_tokens: list[str], b_tokens: list[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    ca = Counter(a_tokens)
    cb = Counter(b_tokens)
    dot = 0.0
    for k, va in ca.items():
        vb = cb.get(k)
        if vb:
            dot += float(va * vb)
    na = math.sqrt(sum(float(v * v) for v in ca.values()))
    nb = math.sqrt(sum(float(v * v) for v in cb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def jaccard(a_tokens: list[str], b_tokens: list[str]) -> float:
    sa = set(a_tokens)
    sb = set(b_tokens)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return max(0.0, min(1.0, inter / union))


def hybrid_similarity(a_text: str, b_text: str) -> float:
    a_norm = normalize_for_similarity(a_text)
    b_norm = normalize_for_similarity(b_text)
    a_toks = tokenize(a_norm)
    b_toks = tokenize(b_norm)
    c = bow_cosine(a_toks, b_toks)
    j = jaccard(a_toks, b_toks)
    # Cosine tends to be more stable; Jaccard helps when lengths differ.
    return max(0.0, min(1.0, 0.75 * c + 0.25 * j))


def order_similarity(a_text: str, b_text: str) -> float:
    """
    Order-aware similarity based on a token sequence alignment ratio.
    This captures "same phrases in the same order" better than BoW.
    """
    a_norm = normalize_for_similarity(a_text)
    b_norm = normalize_for_similarity(b_text)
    a_seq = tokenize_opts(a_norm, remove_stopwords=False)
    b_seq = tokenize_opts(b_norm, remove_stopwords=False)
    if not a_seq or not b_seq:
        return 0.0
    return max(0.0, min(1.0, difflib.SequenceMatcher(a=a_seq, b=b_seq, autojunk=True).ratio()))


def best_evidence_line(reference_text: str, candidate_text: str, *, max_len: int = 160) -> str:
    ref_norm = normalize_for_similarity(reference_text)
    ref_set = set(tokenize(ref_norm))
    if not ref_set:
        return (candidate_text or "").strip().replace("\n", " ")[:max_len]

    best = ""
    best_score = -1
    for line in (candidate_text or "").splitlines():
        ln = line.strip()
        if not ln:
            continue
        ln_norm = normalize_for_similarity(ln)
        ln_set = set(tokenize(ln_norm))
        score = len(ref_set & ln_set)
        if score > best_score:
            best_score = score
            best = ln

    if best:
        best = _WS_RE.sub(" ", best).strip()
        return best[:max_len]

    return (candidate_text or "").strip().replace("\n", " ")[:max_len]
