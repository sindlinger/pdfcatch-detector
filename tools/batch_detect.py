from __future__ import annotations

import sys
import hashlib
import json
import os
import re
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

# Allow `python tools/batch_detect.py` without requiring `PYTHONPATH=src`.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pdfcatch.main import EngineOptions, TemplateRef, score_pdf_against_templates
from Modules.detector.bytes.prefilter import (
    build_model_window_pdf,
    extract_windows_and_fingerprint,
    fingerprint_file,
    pick_best_window_by_tlsh,
    tlsh_distance,
    tlsh_available,
    tlsh_similarity,
)


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _bool(s: str | None, default: bool) -> bool:
    if s is None:
        return default
    v = s.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _int(s: str | None, default: int) -> int:
    if s is None or s.strip() == "":
        return default
    return int(s)


def _float(s: str | None, default: float) -> float:
    if s is None or s.strip() == "":
        return default
    return float(s)


def _resolve(root: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (root / pp).resolve()

def _parse_range_1_based(spec: str) -> tuple[int | None, int | None]:
    """
    Parse a 1-based inclusive range like:
      "10-40" => (10, 40)
      "10"    => (10, 10)
      "-40"   => (None, 40)   (from start)
      "10-"   => (10, None)   (to end)

    Returns (start_1, end_1). Values are validated to be >= 1 if present.
    Empty/whitespace returns (None, None).
    """
    s = (spec or "").strip()
    if not s:
        return (None, None)

    if "-" in s:
        a, b = s.split("-", 1)
        a = a.strip()
        b = b.strip()
        start_1 = int(a) if a else None
        end_1 = int(b) if b else None
    else:
        start_1 = int(s)
        end_1 = int(s)

    if start_1 is not None and start_1 < 1:
        raise ValueError(f"invalid range start (must be >=1): {start_1}")
    if end_1 is not None and end_1 < 1:
        raise ValueError(f"invalid range end (must be >=1): {end_1}")
    if start_1 is not None and end_1 is not None and end_1 < start_1:
        raise ValueError(f"invalid range (end < start): {start_1}-{end_1}")

    return (start_1, end_1)


def _parse_pdf_dir_spec(root: Path, raw: str) -> tuple[Path, int | None, int | None, str | None]:
    """
    Parse PDFCATCH_PDF_DIR, supporting an optional alias+range shorthand:

      PDFCATCH_PDF_DIR=/abs/path
      PDFCATCH_PDF_DIR=relative/path

      PDFCATCH_PDF_DIR=:Q            # alias Q => $PDFCATCH_ALIAS_Q
      PDFCATCH_PDF_DIR=:Q10-40       # alias Q + 1-based range (inclusive)

    Alias mapping:
      PDFCATCH_ALIAS_Q=/mnt/c/.../quarentena

    Returns: (pdf_dir, range_start_1, range_end_1, alias_key)
    """
    s = (raw or "").strip()
    if not s:
        return (_resolve(root, "PDFs"), None, None, None)

    if not s.startswith(":"):
        return (_resolve(root, s), None, None, None)

    m = re.match(r"^:([A-Za-z])(.*)$", s)
    if not m:
        raise ValueError(f"invalid PDFCATCH_PDF_DIR spec: {raw!r}")
    alias = str(m.group(1)).upper()
    rest = (m.group(2) or "").strip()

    alias_path_raw = (os.getenv(f"PDFCATCH_ALIAS_{alias}") or "").strip()
    if not alias_path_raw:
        raise ValueError(f"missing env PDFCATCH_ALIAS_{alias} for PDFCATCH_PDF_DIR={raw!r}")
    pdf_dir = _resolve(root, alias_path_raw)

    start_1, end_1 = _parse_range_1_based(rest) if rest else (None, None)
    return (pdf_dir, start_1, end_1, alias)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _safe_stem(name: str, *, max_len: int = 80) -> str:
    s = _SAFE_NAME_RE.sub("_", name).strip("._-")
    if not s:
        s = "pdf"
    return s[:max_len]


def _load_templates(root: Path, path: Path) -> list[TemplateRef]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("templates") if isinstance(raw, dict) else raw
    if not isinstance(items, list) or not items:
        raise ValueError("templates file must be a JSON list or an object with key 'templates'")

    out: list[TemplateRef] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"template[{i}]: expected object")
        tid = str(it.get("id") or f"tpl_{i}")
        label = str(it.get("label") or "UNKNOWN")
        pdf_raw = it.get("pdf") or it.get("pdf_path") or ""
        pdf_path = _resolve(root, str(pdf_raw)) if str(pdf_raw) else None
        page_index = int(it.get("page_index") or 0)
        if pdf_path is None:
            raise ValueError(f"template[{i}]: missing 'pdf'")
        out.append(TemplateRef(id=tid, label=label, pdf_path=str(pdf_path), page_index=page_index))
    return out


def _summarize(payload: dict[str, Any], *, min_final: float) -> dict[str, Any]:
    pages = payload.get("pages") or []
    templates = payload.get("templates") or []
    labels = sorted({t.get("label") for t in templates if isinstance(t, dict) and t.get("label")})

    best_overall = None
    best_overall_score = -1.0
    for p in pages:
        best = (p or {}).get("best")
        if not best:
            continue
        sc = float(((best.get("scores") or {}).get("final")) or 0.0)
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall = {
                "page_index": int(p.get("page_index")),
                "label": best.get("label"),
                "template_id": best.get("template_id"),
                "final": sc,
            }

    best_by_label: dict[str, Any] = {}
    for lab in labels:
        best = None
        best_score = -1.0
        pages_hit: list[dict[str, Any]] = []
        for p in pages:
            b = (p or {}).get("best")
            if not b or b.get("label") != lab:
                continue
            sc = float(((b.get("scores") or {}).get("final")) or 0.0)
            if sc >= float(min_final):
                pages_hit.append({"page_index": int(p.get("page_index")), "final": sc})
            if sc > best_score:
                best_score = sc
                best = {"page_index": int(p.get("page_index")), "template_id": b.get("template_id"), "final": sc}

        best_by_label[str(lab)] = {
            "best": best if (best and float(best.get("final") or 0.0) >= float(min_final)) else None,
            "hits": sorted(pages_hit, key=lambda r: r["final"], reverse=True)[:20],
        }

    return {
        "candidate": payload.get("candidate"),
        "min_final": float(min_final),
        "best_overall": best_overall if (best_overall and best_overall["final"] >= float(min_final)) else None,
        "best_by_label": best_by_label,
    }

def _aligned_window_score(
    payload: dict[str, Any],
    *,
    template_ids_in_order: list[str],
    window_start_page: int,
) -> dict[str, Any] | None:
    pages = payload.get("pages") or []
    if not isinstance(pages, list):
        return None
    if len(pages) != len(template_ids_in_order):
        return None

    per_page: list[dict[str, Any]] = []
    finals: list[float] = []
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    keys_avg = ["stream", "text_hybrid", "text_order", "image_dhash"]

    for i, tid in enumerate(template_ids_in_order):
        pg = pages[i] or {}
        hit = None
        for h in (pg.get("top") or []):
            if h.get("template_id") == tid:
                hit = h
                break
        if hit is None:
            b = pg.get("best") or {}
            if b.get("template_id") == tid:
                hit = b

        scores = (hit or {}).get("scores") or {}
        try:
            f = float(scores.get("final") or 0.0)
        except Exception:
            f = 0.0
        finals.append(float(f))

        row_scores: dict[str, Any] = {}
        for k in (keys_avg + ["final"]):
            v = scores.get(k)
            row_scores[k] = v
            if k in keys_avg and v is not None:
                try:
                    sums[k] = float(sums.get(k, 0.0)) + float(v)
                    counts[k] = int(counts.get(k, 0)) + 1
                except Exception:
                    pass

        per_page.append(
            {
                "page_index": int(window_start_page) + int(i),
                "template_id": str(tid),
                "scores": row_scores,
            }
        )

    final_avg = float(sum(finals) / float(len(finals) or 1))
    metrics_avg = {k: (float(sums[k]) / float(counts.get(k) or 1)) for k in sums.keys()}
    metrics_avg["final"] = float(final_avg)

    return {"final": float(final_avg), "metrics_avg": metrics_avg, "pages": per_page}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    console = Console()

    # No CLI flag by design: always load configs/run.env from repo root.
    env_path = _resolve(root, "configs/run.env")
    if not env_path.exists():
        console.print(f"[red]error:[/red] env file not found: {env_path}")
        console.print("Create it from configs/run.env.example (copy + edit paths).")
        return 2

    load_dotenv(env_path, override=False)

    pdf_dir_raw = os.getenv("PDFCATCH_PDF_DIR", "PDFs")
    try:
        pdf_dir, range_start_1, range_end_1, alias_key = _parse_pdf_dir_spec(root, str(pdf_dir_raw or "PDFs"))
    except Exception as e:
        console.print(f"[red]error:[/red] invalid PDFCATCH_PDF_DIR: {e}")
        return 2
    recursive = _bool(os.getenv("PDFCATCH_RECURSIVE"), False)
    limit = _int(os.getenv("PDFCATCH_LIMIT"), 10)

    templates_json = _resolve(root, os.getenv("PDFCATCH_TEMPLATES_JSON", "configs/templates.json"))
    if not templates_json.exists():
        fallback = _resolve(root, "configs/templates.example.json")
        if fallback.exists():
            console.print(f"[yellow]warn:[/yellow] templates not found: {templates_json}; using {fallback}")
            templates_json = fallback
        else:
            console.print(f"[red]error:[/red] templates file not found: {templates_json}")
            return 2

    out_dir = _resolve(root, os.getenv("PDFCATCH_OUTPUT_DIR", "Output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    min_final = _float(os.getenv("PDFCATCH_MIN_FINAL"), 0.60)

    page_pref_raw = (os.getenv("PDFCATCH_PAGE_BYTES_PREFILTER_MIN") or "").strip().lower()
    page_pref_min = None if page_pref_raw in {"", "none", "null"} else float(page_pref_raw)
    page_pref_maxdist = _int(os.getenv("PDFCATCH_PAGE_BYTES_PREFILTER_MAXDIST"), 200)
    page_bytes_topn = _int(os.getenv("PDFCATCH_PAGE_BYTES_TOPN"), 10)

    stream_prefilter_raw = os.getenv("PDFCATCH_STREAM_PREFILTER_MIN", "0.45")
    if stream_prefilter_raw is None:
        stream_prefilter_min = 0.45
    else:
        v = stream_prefilter_raw.strip().lower()
        stream_prefilter_min = None if v in {"", "none", "null"} else float(stream_prefilter_raw)

    opts = EngineOptions(
        top_k=_int(os.getenv("PDFCATCH_TOP_K"), 3),
        enable_stream=_bool(os.getenv("PDFCATCH_ENABLE_STREAM"), True),
        enable_text=_bool(os.getenv("PDFCATCH_ENABLE_TEXT"), True),
        enable_image=_bool(os.getenv("PDFCATCH_ENABLE_IMAGE"), True),
        image_dpi=_int(os.getenv("PDFCATCH_IMAGE_DPI"), 72),
        image_hash_size=_int(os.getenv("PDFCATCH_IMAGE_HASH_SIZE"), 8),
        stream_prefilter_min=stream_prefilter_min,
    )

    if not pdf_dir.exists() or not pdf_dir.is_dir():
        console.print(f"[red]error:[/red] PDF directory not found: {pdf_dir}")
        return 2

    # Collect PDFs.
    if recursive:
        candidates = [p for p in pdf_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
    else:
        candidates = [p for p in pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    candidates.sort(key=lambda p: str(p).lower())

    # Optional 1-based selection range (inclusive). Applied before PDFCATCH_LIMIT.
    if range_start_1 is not None or range_end_1 is not None:
        total = len(candidates)
        s0 = (int(range_start_1) - 1) if range_start_1 is not None else 0
        e0 = int(range_end_1) if range_end_1 is not None else total
        s0 = max(0, min(s0, total))
        e0 = max(0, min(e0, total))
        candidates = candidates[s0:e0]
        console.print(
            f"[dim]PDF selection:[/dim] dir={pdf_dir} alias={alias_key or '-'} range="
            f"{range_start_1 or ''}-{range_end_1 or ''} => {len(candidates)}/{total} PDFs"
        )
    elif limit > 0:
        candidates = candidates[:limit]

    if not candidates:
        console.print(f"[yellow]warn:[/yellow] no PDFs found in: {pdf_dir}")
        return 0

    templates = _load_templates(root, templates_json)

    target_label = (os.getenv("PDFCATCH_TARGET_LABEL") or "").strip()
    labels_in_templates = sorted({t.label for t in templates})
    if not target_label and len(labels_in_templates) > 1:
        console.print("[red]error:[/red] multiple labels found in templates.")
        console.print("Set PDFCATCH_TARGET_LABEL in configs/run.env to one of:")
        for lab in labels_in_templates:
            console.print(f"  - {lab}")
        return 2
    if target_label:
        templates = [t for t in templates if t.label == target_label]
        if not templates:
            console.print(f"[red]error:[/red] no templates for label: {target_label}")
            return 2

    templates_seq = sorted(templates, key=lambda t: int(t.page_index))
    K = int(len(templates_seq))
    tpl_meta = [{"template_id": t.id, "pdf": t.pdf_path, "page_index": int(t.page_index)} for t in templates_seq]
    label = str(templates_seq[0].label) if templates_seq else "UNKNOWN"
    tids = [t.id for t in templates_seq]

    # Stage 0 prep: build a model window PDF (K pages) and hash it as an opaque file.
    model_dir = (out_dir / "_model_windows" / label).resolve()
    model_pdf = build_model_window_pdf(templates=tpl_meta, window_len=K, out_dir=model_dir)
    model_fp = fingerprint_file(model_pdf)
    model_size_bytes = int(model_fp.size_bytes)
    model_tlsh = model_fp.tlsh

    def best_overall_for_sequence(payload: dict[str, Any]) -> dict[str, Any] | None:
        pages = payload.get("pages") or []
        cand = payload.get("candidate") or {}
        page_count = int(cand.get("page_count") or 0)
        if K <= 0 or page_count <= 0 or len(pages) != page_count:
            return None

        per_page: list[dict[str, float]] = []
        for pg in pages:
            m: dict[str, float] = {}
            for hit in (pg.get("top") or []):
                try:
                    tid = str(hit.get("template_id") or "")
                    sc = float(((hit.get("scores") or {}).get("final")) or 0.0)
                except Exception:
                    continue
                if tid:
                    m[tid] = sc
            per_page.append(m)

        # Label already computed above.

        if K == 1:
            tid = tids[0]
            best_page = -1
            best_final = -1.0
            for i in range(page_count):
                sc = float(per_page[i].get(tid) or 0.0)
                if sc > best_final:
                    best_final = sc
                    best_page = i
            if best_page < 0:
                return None
            return {"label": label, "page_index": int(best_page), "template_id": tid, "final": float(best_final)}

        if page_count < K:
            return None

        best_start = -1
        best_final = -1.0
        for start in range(0, page_count - K + 1):
            finals: list[float] = []
            for pos, tid in enumerate(tids):
                finals.append(float(per_page[start + pos].get(tid) or 0.0))
            sc = float(sum(finals) / float(K))
            if sc > best_final:
                best_final = sc
                best_start = start

        if best_start < 0:
            return None
        return {
            "label": label,
            "start_page": int(best_start),
            "end_page": int(best_start + K - 1),
            "template_ids": tids,
            "final": float(best_final),
        }

    # Index of summaries for quick review.
    index: list[dict[str, Any]] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        pdf_task = progress.add_task("PDFs", total=len(candidates))
        page_task = progress.add_task("Pages", total=0, visible=False)

        for pdf_path in candidates:
            pdf_sha = _sha256_file(pdf_path)
            run_name = f"{_safe_stem(pdf_path.stem)}__{pdf_sha[:12]}"
            pdf_out = out_dir / run_name
            pdf_out.mkdir(parents=True, exist_ok=True)

            progress.update(pdf_task, description=f"PDFs (current: {pdf_path.name})")
            progress.update(page_task, visible=True, total=1, completed=0, description="Pages")

            def on_event(ev: dict[str, Any]) -> None:
                typ = ev.get("type")
                if typ == "open_candidate":
                    total_pages = int(ev.get("page_count") or 0)
                    progress.update(page_task, total=max(1, total_pages), completed=0, description=f"Pages ({pdf_path.name})")
                elif typ == "page_done":
                    # We set completed based on page_index to be stable.
                    pi = int(ev.get("page_index") or 0)
                    progress.update(page_task, completed=pi + 1)

            # Stage 0: extract ALL K-page windows to disk, hash by opaque file bytes (TLSH), pick best window.
            windows_dir = (pdf_out / f"windows_k{K}").resolve()
            windows_info = extract_windows_and_fingerprint(
                candidate_pdf=pdf_path,
                window_len=K,
                out_dir=windows_dir,
                step=1,
                overwrite=False,
            )
            (windows_dir / "index.json").write_text(
                json.dumps(windows_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

            ranked_windows: list[dict[str, Any]] = []
            if tlsh_available() and model_tlsh:
                for w in (windows_info.get("windows") or []):
                    fp = (w.get("fingerprint") or {}) if isinstance(w, dict) else {}
                    wsize = fp.get("size_bytes")
                    try:
                        wsize_i = int(wsize) if wsize is not None else None
                    except Exception:
                        wsize_i = None

                    size_similarity = None
                    size_delta_bytes = None
                    if model_size_bytes > 0 and wsize_i and wsize_i > 0:
                        size_similarity = float(min(model_size_bytes, wsize_i) / max(model_size_bytes, wsize_i))
                        size_delta_bytes = int(abs(model_size_bytes - wsize_i))
                    wt = fp.get("tlsh")
                    sim = tlsh_similarity(wt, model_tlsh, max_dist=int(page_pref_maxdist))
                    dist = tlsh_distance(wt, model_tlsh)
                    if sim is None or dist is None:
                        continue
                    ranked_windows.append(
                        {
                            "start_page": int(w.get("start_page") or 0),
                            "end_page": int(w.get("end_page") or 0),
                            "window_pdf": str(w.get("pdf") or ""),
                            "size_bytes": wsize_i,
                            "size_similarity": size_similarity,
                            "size_delta_bytes": size_delta_bytes,
                            "similarity": float(sim),
                            "distance": int(dist),
                        }
                    )
                ranked_windows.sort(key=lambda r: r.get("similarity", 0.0), reverse=True)

            best = pick_best_window_by_tlsh(
                windows=list(windows_info.get("windows") or []),
                model_tlsh=str(model_tlsh or ""),
                max_dist=int(page_pref_maxdist),
            )
            page_pref = {
                "method": "window_file_tlsh",
                "available": bool(tlsh_available()) and bool(model_tlsh),
                "window_len": int(K),
                "topn": int(page_bytes_topn),
                "max_dist": int(page_pref_maxdist),
                "model": {
                    "pdf": str(model_pdf),
                    "fingerprint": {
                        "size_bytes": int(model_fp.size_bytes),
                        "sha256": str(model_fp.sha256),
                        "tlsh": model_fp.tlsh,
                    },
                },
                "candidate": {
                    "pdf": str(pdf_path),
                    "page_count": int((windows_info.get("candidate") or {}).get("page_count") or 0),
                    "windows_dir": str(windows_dir),
                    "windows_index": str((windows_dir / "index.json").resolve()),
                    "window_count": int(len(windows_info.get("windows") or [])),
                },
                "best": best,
                "top_windows": ranked_windows[: max(0, int(page_bytes_topn))] if int(page_bytes_topn) > 0 else [],
            }
            if isinstance(page_pref.get("best"), dict):
                bs = page_pref["best"].get("candidate_size_bytes")
                try:
                    bs_i = int(bs) if bs is not None else None
                except Exception:
                    bs_i = None
                if model_size_bytes > 0 and bs_i and bs_i > 0:
                    page_pref["best"]["size_similarity"] = float(min(model_size_bytes, bs_i) / max(model_size_bytes, bs_i))
                    page_pref["best"]["size_delta_bytes"] = int(abs(model_size_bytes - bs_i))
            best_pref = (page_pref.get("best") or {}) if isinstance(page_pref, dict) else {}
            pref_sim = best_pref.get("similarity")
            if (
                page_pref_min is not None
                and bool(page_pref.get("available"))
                and pref_sim is not None
                and float(pref_sim) < float(page_pref_min)
            ):
                (pdf_out / "prefilter.json").write_text(
                    json.dumps(page_pref, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                index.append(
                    {
                        "pdf": str(pdf_path),
                        "sha256": pdf_sha,
                        "out_dir": str(pdf_out),
                        "best_overall": None,
                        "page_prefilter": page_pref,
                        "error": "skipped_by_page_prefilter",
                    }
                )
                progress.update(page_task, visible=False)
                progress.advance(pdf_task, 1)
                continue

            try:
                # Ensure top_k keeps all templates so we can compute aligned window scores reliably.
                opts_run = opts
                if int(opts.top_k) < K:
                    opts_run = EngineOptions(
                        enable_stream=opts.enable_stream,
                        stream_prefilter_min=opts.stream_prefilter_min,
                        enable_text=opts.enable_text,
                        enable_image=opts.enable_image,
                        image_dpi=opts.image_dpi,
                        image_hash_size=opts.image_hash_size,
                        top_k=K,
                        w_text_hybrid=opts.w_text_hybrid,
                        w_text_order=opts.w_text_order,
                        w_image_dhash=opts.w_image_dhash,
                    )

                windows_eval: list[dict[str, Any]] = []
                if bool(page_pref.get("available")) and int(page_bytes_topn) > 0 and ranked_windows:
                    eval_dir = (pdf_out / "window_results").resolve()
                    eval_dir.mkdir(parents=True, exist_ok=True)
                    selected = ranked_windows[: max(1, int(page_bytes_topn))]
                    for rw in selected:
                        win_pdf = str(rw.get("window_pdf") or "")
                        if not win_pdf:
                            continue
                        payload_win = score_pdf_against_templates(win_pdf, templates_seq, options=opts_run, on_event=on_event)
                        aligned = _aligned_window_score(
                            payload_win,
                            template_ids_in_order=tids,
                            window_start_page=int(rw.get("start_page") or 0),
                        )
                        out_json = eval_dir / f"win_p{int(rw.get('start_page') or 0):04d}-{int(rw.get('end_page') or 0):04d}.json"
                        out_json.write_text(json.dumps(payload_win, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                        windows_eval.append(
                            {
                                "start_page": int(rw.get("start_page") or 0),
                                "end_page": int(rw.get("end_page") or 0),
                                "window_pdf": win_pdf,
                                "stage0": {
                                    "similarity": float(rw.get("similarity") or 0.0),
                                    "distance": int(rw.get("distance") or 0),
                                    "size_bytes": rw.get("size_bytes"),
                                    "size_similarity": rw.get("size_similarity"),
                                    "size_delta_bytes": rw.get("size_delta_bytes"),
                                },
                                "aligned": aligned,
                                "result_json": str(out_json),
                            }
                        )

                best_overall = None
                best_payload_path = None
                if windows_eval:
                    # pick best by aligned.final
                    def _win_final(w: dict[str, Any]) -> float:
                        a = w.get("aligned") or {}
                        try:
                            return float(a.get("final") or 0.0)
                        except Exception:
                            return 0.0

                    wbest = max(windows_eval, key=_win_final)
                    a = wbest.get("aligned") or {}
                    stage0 = wbest.get("stage0") or {}
                    best_overall = {
                        "label": label,
                        "start_page": int(wbest.get("start_page") or 0),
                        "end_page": int(wbest.get("end_page") or 0),
                        "template_ids": tids,
                        "final": float(a.get("final") or 0.0),
                        "stage0_similarity": float(stage0.get("similarity") or 0.0),
                        "stage0_distance": int(stage0.get("distance") or 0),
                        "stage0_size_similarity": stage0.get("size_similarity"),
                        "stage0_size_delta_bytes": stage0.get("size_delta_bytes"),
                        "window_pdf": str(wbest.get("window_pdf") or ""),
                    }
                    best_payload_path = str(wbest.get("result_json") or "")
                    # For convenience/compat: always expose the best window payload as result.json.
                    if best_payload_path:
                        try:
                            src_p = Path(best_payload_path)
                            if src_p.exists():
                                (pdf_out / "result.json").write_bytes(src_p.read_bytes())
                                best_payload_path = str((pdf_out / "result.json").resolve())
                        except Exception:
                            pass
                else:
                    # Fallback: full scan when TLSH is unavailable (or topn==0).
                    payload = score_pdf_against_templates(str(pdf_path), templates_seq, options=opts_run, on_event=on_event)
                    best_overall = best_overall_for_sequence(payload)
                    (pdf_out / "result.json").write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                    )

                summary = {
                    "candidate_pdf": str(pdf_path),
                    "sha256": pdf_sha,
                    "label": label,
                    "min_final": float(min_final),
                    "best_overall": best_overall,
                    "page_prefilter": page_pref,
                    "windows_eval": windows_eval,
                    "target_label": target_label or None,
                    "best_payload_json": best_payload_path,
                }

                (pdf_out / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                (pdf_out / "prefilter.json").write_text(
                    json.dumps(page_pref, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                (pdf_out / "windows_eval.json").write_text(
                    json.dumps(windows_eval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )

                index.append(
                    {
                        "pdf": str(pdf_path),
                        "sha256": pdf_sha,
                        "out_dir": str(pdf_out),
                        "best_overall": best_overall,
                        "page_prefilter": page_pref,
                        "error": None,
                    }
                )

            except Exception as e:
                (pdf_out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                index.append(
                    {
                        "pdf": str(pdf_path),
                        "sha256": pdf_sha,
                        "out_dir": str(pdf_out),
                        "best_overall": None,
                        "page_prefilter": page_pref,
                        "error": str(e),
                    }
                )

            progress.update(page_task, visible=False)
            progress.advance(pdf_task, 1)

    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    console.print(f"wrote: {out_dir / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
