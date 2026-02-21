from __future__ import annotations

import hashlib
import re


_WS_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _hash64(token: str) -> int:
    # Stable 64-bit hash.
    h = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(h, byteorder="little", signed=False)


def simhash64(tokens: list[str]) -> int:
    """
    Classic 64-bit SimHash over tokens.
    """
    if not tokens:
        return 0

    # Bit weights (int) to avoid floating error.
    w = [0] * 64
    for t in tokens:
        hv = _hash64(t)
        for i in range(64):
            if (hv >> i) & 1:
                w[i] += 1
            else:
                w[i] -= 1

    out = 0
    for i, wi in enumerate(w):
        if wi > 0:
            out |= 1 << i
    return int(out)


def tokenize_pdf_ops(s: str) -> list[str]:
    """
    Tokenize decoded PDF content stream text.

    This is intentionally simple and robust:
    - keep alnum tokens
    - normalize digits to '#'
    """
    if not s:
        return []
    s = s.lower()
    s = _DIGIT_RE.sub("#", s)
    s = _WS_RE.sub(" ", s).strip()
    toks: list[str] = []
    for m in _TOKEN_RE.finditer(s):
        t = m.group(0)
        if len(t) <= 1:
            continue
        toks.append(t)
    return toks


def simhash64_from_bytes(data: bytes) -> int:
    if not data:
        return 0
    # Latin-1 keeps a 1:1 mapping for bytes 0..255; avoids decode failures.
    s = data.decode("latin-1", errors="ignore")
    toks = tokenize_pdf_ops(s)
    return simhash64(toks)


def hamming_distance64(a: int, b: int) -> int:
    return int((int(a) ^ int(b)).bit_count())


def simhash_similarity(a: int, b: int) -> float:
    """
    0..1 similarity based on 64-bit Hamming distance.
    """
    dist = hamming_distance64(a, b)
    return float(max(0.0, min(1.0, 1.0 - (dist / 64.0))))

