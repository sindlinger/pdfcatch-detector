from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemplateRef:
    """
    Reference template page.

    `id` is an opaque stable identifier (caller-controlled).
    """

    id: str
    label: str
    pdf_path: str
    page_index: int


@dataclass(frozen=True)
class EngineOptions:
    # Stream fingerprinting (decoded /Contents).
    enable_stream: bool = True
    stream_prefilter_min: float | None = 0.45  # if set, may skip expensive steps when below

    # Text similarity (requires text layer; otherwise text metrics drop to ~0).
    enable_text: bool = True

    # Image similarity (render + dHash). Expensive relative to text/streams.
    enable_image: bool = True
    image_dpi: int = 72
    image_hash_size: int = 8

    # Output control.
    top_k: int = 5

    # Default final-score weights (only applied when a metric is available).
    w_text_hybrid: float = 0.55
    w_text_order: float = 0.20
    w_image_dhash: float = 0.25


@dataclass(frozen=True)
class MetricScores:
    # 0..1 scores; None means metric not computed/unavailable.
    stream: float | None
    text_cosine: float | None
    text_jaccard: float | None
    text_hybrid: float | None
    text_order: float | None
    image_dhash: float | None
    final: float


@dataclass(frozen=True)
class TemplateHit:
    template_id: str
    label: str
    scores: MetricScores


@dataclass(frozen=True)
class PageResult:
    page_index: int
    best: TemplateHit | None
    top: list[TemplateHit]
