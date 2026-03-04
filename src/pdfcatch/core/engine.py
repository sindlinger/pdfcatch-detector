from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Any

import fitz  # PyMuPDF

from pdfcatch.core.model import EngineOptions, MetricScores, PageResult, TemplateHit, TemplateRef


def _final_score(
    opts: EngineOptions,
    *,
    text_hybrid: float | None,
    text_order: float | None,
    image_dhash: float | None,
) -> float:
    parts: list[tuple[float, float]] = []
    if text_hybrid is not None:
        parts.append((opts.w_text_hybrid, text_hybrid))
    if text_order is not None:
        parts.append((opts.w_text_order, text_order))
    if image_dhash is not None:
        parts.append((opts.w_image_dhash, image_dhash))

    if not parts:
        return 0.0
    wsum = sum(w for w, _ in parts)
    if wsum <= 0:
        return 0.0
    return float(sum(w * v for w, v in parts) / wsum)


def _safe_open_fitz(path: str | Path) -> fitz.Document:
    return fitz.open(str(path))


def score_pdf_against_templates(
    candidate_pdf: str | Path,
    templates: list[TemplateRef],
    *,
    options: EngineOptions | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> dict:
    """
    Compute per-page similarity between `candidate_pdf` and a set of template pages.

    This is a library function (no CLI). Returns a JSON-serializable dict.
    """
    opts = options or EngineOptions()
    cand_path = Path(candidate_pdf)

    if not cand_path.exists():
        raise FileNotFoundError(str(cand_path))
    if not templates:
        raise ValueError("templates: empty")

    def emit(typ: str, **kwargs: Any) -> None:
        if on_event is None:
            return
        try:
            on_event({"type": typ, **kwargs})
        except Exception:
            # UI callbacks must not break the engine.
            return

    emit("start", candidate_pdf=str(cand_path), template_count=len(templates), options=asdict(opts))

    # Open candidate docs once.
    cand_fitz = _safe_open_fitz(cand_path)
    cand_pike = open_pikepdf(cand_path) if opts.enable_stream else None

    # Open templates (group by PDF to avoid reopening per template page).
    tpl_by_pdf: dict[str, list[TemplateRef]] = {}
    for t in templates:
        tpl_by_pdf.setdefault(str(Path(t.pdf_path)), []).append(t)

    tpl_fitz_docs: dict[str, fitz.Document] = {}
    tpl_pike_docs: dict[str, object] = {}
    try:
        emit("open_candidate", page_count=int(cand_fitz.page_count))
        for pdfp in tpl_by_pdf:
            tpl_fitz_docs[pdfp] = _safe_open_fitz(pdfp)
            if opts.enable_stream:
                tpl_pike_docs[pdfp] = open_pikepdf(pdfp)

        # Precompute template features.
        emit("templates_precompute_start", template_count=len(templates), pdf_count=len(tpl_by_pdf))
        tpl_feats: dict[str, dict] = {}
        for pdfp, tpls in tpl_by_pdf.items():
            fdoc = tpl_fitz_docs[pdfp]
            pdoc = tpl_pike_docs.get(pdfp)
            for t in tpls:
                if not (0 <= t.page_index < fdoc.page_count):
                    raise ValueError(f"template page out of range: {t.id} {t.pdf_path}#{t.page_index}")
                txt = fdoc.load_page(t.page_index).get_text("text") or ""
                dh = None
                if opts.enable_image:
                    dh = page_dhash(
                        fdoc, page_index=t.page_index, hash_size=opts.image_hash_size, dpi=opts.image_dpi
                    )

                stream_fp = None
                if opts.enable_stream and pdoc is not None:
                    stream_fp = fingerprint_page_contents(pdoc, page_index=t.page_index)

                tpl_feats[t.id] = {
                    "template": t,
                    "text": txt,
                    "dhash": dh,
                    "stream_fp": stream_fp,
                }
        emit("templates_precompute_done", template_count=len(tpl_feats))

        results: list[PageResult] = []
        for page_index in range(cand_fitz.page_count):
            emit("page_start", page_index=int(page_index), page_count=int(cand_fitz.page_count))
            cand_text = cand_fitz.load_page(page_index).get_text("text") or ""

            cand_stream_fp = None
            if opts.enable_stream and cand_pike is not None:
                cand_stream_fp = fingerprint_page_contents(cand_pike, page_index=page_index)
            emit("page_features", page_index=int(page_index), have_stream=bool(cand_stream_fp), text_chars=len(cand_text))

            # Score against each template.
            hits: list[TemplateHit] = []
            cand_dh = None
            cand_dh_ready = False
            for tid, feat in tpl_feats.items():
                t: TemplateRef = feat["template"]

                # 1) stream prefilter
                stream_score = None
                if opts.enable_stream and cand_stream_fp is not None and feat["stream_fp"] is not None:
                    stream_score = stream_similarity(cand_stream_fp, feat["stream_fp"])

                # 2) text scores
                text_scores = None
                if opts.enable_text:
                    text_scores = score_text(feat["text"], cand_text)

                # 3) image scores (optionally skipped if prefilter suggests very different)
                image_score = None
                can_compute_image = bool(opts.enable_image and feat["dhash"] is not None)
                if can_compute_image:
                    if opts.stream_prefilter_min is not None and stream_score is not None:
                        if stream_score < float(opts.stream_prefilter_min):
                            can_compute_image = False
                if can_compute_image:
                    if not cand_dh_ready:
                        cand_dh = page_dhash(
                            cand_fitz, page_index=page_index, hash_size=opts.image_hash_size, dpi=opts.image_dpi
                        )
                        cand_dh_ready = True
                    if cand_dh is not None:
                        image_score = dhash_score(feat["dhash"], cand_dh)

                final = _final_score(
                    opts,
                    text_hybrid=(text_scores.hybrid if text_scores else None),
                    text_order=(text_scores.order if text_scores else None),
                    image_dhash=image_score,
                )

                scores = MetricScores(
                    stream=stream_score,
                    text_cosine=(text_scores.cosine if text_scores else None),
                    text_jaccard=(text_scores.jaccard if text_scores else None),
                    text_hybrid=(text_scores.hybrid if text_scores else None),
                    text_order=(text_scores.order if text_scores else None),
                    image_dhash=image_score,
                    final=float(final),
                )
                hits.append(TemplateHit(template_id=t.id, label=t.label, scores=scores))

            hits.sort(key=lambda m: m.scores.final, reverse=True)
            top_k = max(1, min(int(opts.top_k), len(hits)))
            top = hits[:top_k]
            best = top[0] if top else None
            results.append(PageResult(page_index=int(page_index), best=best, top=top))
            emit(
                "page_done",
                page_index=int(page_index),
                best=(best.label if best else None),
                best_score=(best.scores.final if best else None),
                top_k=top_k,
                top=[
                    {
                        "template_id": m.template_id,
                        "label": m.label,
                        "final": m.scores.final,
                        "stream": m.scores.stream,
                        "text_hybrid": m.scores.text_hybrid,
                        "text_order": m.scores.text_order,
                        "image_dhash": m.scores.image_dhash,
                    }
                    for m in top
                ],
            )

        payload = {
            "candidate": {
                "pdf": str(cand_path),
                "page_count": int(cand_fitz.page_count),
            },
            "templates": [
                {"id": t.id, "label": t.label, "pdf": str(Path(t.pdf_path)), "page_index": int(t.page_index)}
                for t in templates
            ],
            "options": asdict(opts),
            "pages": [
                {
                    "page_index": pr.page_index,
                    "best": (
                        {
                            "template_id": pr.best.template_id,
                            "label": pr.best.label,
                            "scores": asdict(pr.best.scores),
                        }
                        if pr.best
                        else None
                    ),
                    "top": [
                        {
                            "template_id": m.template_id,
                            "label": m.label,
                            "scores": asdict(m.scores),
                        }
                        for m in pr.top
                    ],
                }
                for pr in results
            ],
        }
        emit("done", page_count=int(cand_fitz.page_count))
        return payload
    finally:
        try:
            cand_fitz.close()
        except Exception:
            pass
        if cand_pike is not None:
            try:
                cand_pike.close()
            except Exception:
                pass
        for d in tpl_fitz_docs.values():
            try:
                d.close()
            except Exception:
                pass
        for d in tpl_pike_docs.values():
            try:
                d.close()
            except Exception:
                pass
