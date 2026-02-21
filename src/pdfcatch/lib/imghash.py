from __future__ import annotations

from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass(frozen=True)
class DHash:
    value: int
    bits: int
    hash_size: int
    dpi: int

    def hex(self) -> str:
        # Pad to full nybbles for stable output.
        width = (self.bits + 3) // 4
        return f"{self.value:0{width}x}"


def dhash_from_gray_bytes(samples: bytes, width: int, height: int, *, hash_size: int) -> int:
    """
    Compute a simple dHash (difference hash) over a grayscale image.

    - `samples` must be row-major 8-bit grayscale (len == width*height).
    - returns an integer bitset of `hash_size*hash_size` bits.
    """
    if width <= 0 or height <= 0:
        return 0
    if hash_size <= 0:
        return 0

    tgt_w = hash_size + 1
    tgt_h = hash_size

    xs = [(x * width) // tgt_w for x in range(tgt_w)]
    ys = [(y * height) // tgt_h for y in range(tgt_h)]

    out = 0
    bit = 0
    for y in ys:
        row_off = y * width
        for x in range(hash_size):
            a = samples[row_off + xs[x]]
            b = samples[row_off + xs[x + 1]]
            if b > a:
                out |= 1 << bit
            bit += 1
    return out


def dhash_from_page(page: fitz.Page, *, hash_size: int = 8, dpi: int = 72) -> DHash | None:
    try:
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY, alpha=False)
    except Exception:
        return None

    if pix.width <= 0 or pix.height <= 0:
        return None
    if not pix.samples:
        return None

    v = dhash_from_gray_bytes(pix.samples, pix.width, pix.height, hash_size=hash_size)
    bits = int(hash_size * hash_size)
    return DHash(value=int(v), bits=bits, hash_size=int(hash_size), dpi=int(dpi))


def hamming_distance(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def dhash_similarity(a: DHash, b: DHash) -> tuple[float, int]:
    """
    Returns (similarity 0..1, distance bits).
    """
    if a.bits != b.bits:
        # Different sizes: compare on the smallest bit-width by truncating.
        bits = min(a.bits, b.bits)
        mask = (1 << bits) - 1
        dist = hamming_distance(a.value & mask, b.value & mask)
        score = 1.0 - (dist / float(bits)) if bits else 0.0
        return max(0.0, min(1.0, score)), int(dist)

    dist = hamming_distance(a.value, b.value)
    score = 1.0 - (dist / float(a.bits)) if a.bits else 0.0
    return max(0.0, min(1.0, score)), int(dist)

