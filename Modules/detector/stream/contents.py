from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pikepdf

from Modules.detector.stream.simhash64 import simhash64_from_bytes, simhash_similarity


@dataclass(frozen=True)
class PageContentsFingerprint:
    """
    Fingerprints computed over decoded page /Contents streams.
    """

    stream_sizes: list[int]
    sha256_concat: str
    simhash_concat: int
    simhash_largest: int


def _iter_contents_streams(page_obj: pikepdf.Object) -> Iterable[pikepdf.Stream]:
    contents = page_obj.get("/Contents")
    if contents is None:
        return

    # pikepdf returns a pikepdf.objects.Object wrapper for arrays and streams.
    if isinstance(contents, pikepdf.Array):
        for item in contents:
            if isinstance(item, pikepdf.Stream):
                yield item
    elif isinstance(contents, pikepdf.Stream):
        yield contents


def page_contents_bytes(pdf: pikepdf.Pdf, *, page_index: int) -> list[bytes]:
    page = pdf.pages[page_index]
    out: list[bytes] = []
    for st in _iter_contents_streams(page.obj):
        try:
            out.append(st.read_bytes())  # decoded bytes
        except Exception:
            # Best-effort: ignore unreadable streams.
            continue
    return out


def fingerprint_page_contents(pdf: pikepdf.Pdf, *, page_index: int) -> PageContentsFingerprint:
    streams = page_contents_bytes(pdf, page_index=page_index)
    sizes = [len(b) for b in streams]
    concat = b"".join(streams)
    sha = hashlib.sha256(concat).hexdigest()
    sh_concat = simhash64_from_bytes(concat)

    if streams:
        largest = max(streams, key=len)
        sh_largest = simhash64_from_bytes(largest)
    else:
        sh_largest = 0

    return PageContentsFingerprint(
        stream_sizes=sizes,
        sha256_concat=sha,
        simhash_concat=int(sh_concat),
        simhash_largest=int(sh_largest),
    )


def stream_similarity(a: PageContentsFingerprint, b: PageContentsFingerprint) -> float:
    """
    Stream-based similarity (0..1) using simhash over:
    - concatenation of streams
    - largest stream

    We take the max of both, because some pages have small prolog/epilog streams
    plus one main stream; comparing only concat can be slightly noisier.
    """
    s1 = simhash_similarity(a.simhash_concat, b.simhash_concat)
    s2 = simhash_similarity(a.simhash_largest, b.simhash_largest)
    return float(max(s1, s2))


def open_pdf(path: str | Path) -> pikepdf.Pdf:
    return pikepdf.open(str(path))

