from __future__ import annotations

"""
Stage-0 triage based on **opaque file bytes** (no `/Contents` decoding).

Workflow (DESPACHO example, K=2):
1) Build a *model window PDF* containing exactly K pages (in order) from the active templates.
2) For each candidate PDF, extract *all* K-page windows to disk as standalone PDFs.
3) Compute TLSH on the **bytes of each extracted window file** (treat as an opaque file).
4) Compare each candidate window TLSH against the model window TLSH; best window wins.

Important:
- We DO open PDFs with PyMuPDF to *extract* pages into new window-PDF files (required work).
- We DO NOT open/inspect the extracted window PDFs to read `/Contents` or operators.
  Hashing is done by reading raw file bytes.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


try:
    import tlsh as _tlsh  # provided by the `py-tlsh` package
except Exception:  # pragma: no cover
    _tlsh = None


# Make extracted mini-PDFs stable-ish so caching works across runs.
_SAVE_KW = dict(
    garbage=4,
    clean=True,
    deflate=True,
    no_new_id=True,
    preserve_metadata=0,
)


@dataclass(frozen=True)
class FileFingerprint:
    size_bytes: int
    sha256: str
    tlsh: str | None


def tlsh_available() -> bool:
    return _tlsh is not None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tlsh_hash_bytes(data: bytes) -> str | None:
    if _tlsh is None:
        return None
    try:
        h = _tlsh.hash(data)
    except Exception:
        return None
    if not h or h == "TNULL":
        return None
    return str(h)


def fingerprint_file(path: str | Path) -> FileFingerprint:
    """
    Compute hashes by reading the file bytes only (no PDF parsing/inspection).
    """
    p = Path(path)
    data = p.read_bytes()
    return FileFingerprint(size_bytes=int(len(data)), sha256=_sha256_bytes(data), tlsh=_tlsh_hash_bytes(data))


def tlsh_distance(a: str | None, b: str | None) -> int | None:
    if _tlsh is None:
        return None
    if not a or not b:
        return None
    try:
        return int(_tlsh.diffxlen(str(a), str(b)))
    except Exception:
        return None


def tlsh_similarity(a: str | None, b: str | None, *, max_dist: int = 200) -> float | None:
    d = tlsh_distance(a, b)
    if d is None:
        return None
    md = max(1, int(max_dist))
    return float(max(0.0, min(1.0, 1.0 - (float(d) / float(md)))))


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def extract_window_pdf_file(
    *,
    src_doc: fitz.Document,
    start_page: int,
    end_page: int,
    out_path: Path,
) -> None:
    """
    Extract pages [start_page..end_page] (inclusive) as a standalone PDF file on disk.
    """
    out = fitz.open()
    try:
        out.insert_pdf(src_doc, from_page=int(start_page), to_page=int(end_page))
        data = out.tobytes(**_SAVE_KW)
    finally:
        try:
            out.close()
        except Exception:
            pass
    _write_bytes(out_path, data)


def build_model_window_pdf(
    *,
    templates: list[dict[str, Any]],
    window_len: int,
    out_dir: str | Path,
) -> Path:
    """
    Build a model PDF containing exactly K pages (in order) from `templates[:K]`.

    `templates` items must contain: {"pdf": ..., "page_index": ...} in order.
    """
    K = int(window_len)
    if K <= 0:
        raise ValueError("window_len must be >= 1")
    if len(templates) < K:
        raise ValueError("templates: expected at least window_len items")

    # Signature for caching: resolved PDF paths + page indexes + template ids.
    sig_items: list[dict[str, Any]] = []
    for t in templates[:K]:
        pdfp = str(Path(str(t.get("pdf") or "")).resolve())
        sig_items.append(
            {
                "template_id": str(t.get("template_id") or t.get("id") or ""),
                "pdf": pdfp,
                "page_index": int(t.get("page_index") or 0),
            }
        )
    sig = hashlib.sha256(json.dumps(sig_items, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    model_path = (out_dir_p / f"model__{sig}__k{K}.pdf").resolve()
    if model_path.exists():
        return model_path

    tpl_docs: dict[str, fitz.Document] = {}
    out = fitz.open()
    try:
        # Open required template PDFs once.
        for t in templates[:K]:
            pdfp = str(Path(str(t.get("pdf") or "")).resolve())
            if pdfp and pdfp not in tpl_docs:
                tpl_docs[pdfp] = fitz.open(pdfp)

        # Insert pages in order.
        for t in templates[:K]:
            pdfp = str(Path(str(t.get("pdf") or "")).resolve())
            doc = tpl_docs.get(pdfp)
            if doc is None:
                raise FileNotFoundError(pdfp)
            pi = int(t.get("page_index") or 0)
            out.insert_pdf(doc, from_page=int(pi), to_page=int(pi))

        data = out.tobytes(**_SAVE_KW)
        _write_bytes(model_path, data)
        return model_path
    finally:
        try:
            out.close()
        except Exception:
            pass
        for d in tpl_docs.values():
            try:
                d.close()
            except Exception:
                pass


def extract_windows_and_fingerprint(
    *,
    candidate_pdf: str | Path,
    window_len: int,
    out_dir: str | Path,
    step: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Extract all K-page windows from a candidate PDF to disk, and fingerprint each window file.

    Returns a JSON-serializable dict:
      {
        "candidate": {"pdf":..., "page_count": N},
        "window_len": K,
        "step": 1,
        "windows": [
          {"start_page": i, "end_page": j, "pdf": ".../win_p0000-0001.pdf",
           "fingerprint": {"size_bytes":..,"sha256":..,"tlsh":..}}
        ]
      }
    """
    cand_path = Path(candidate_pdf).resolve()
    K = int(window_len)
    if K <= 0:
        raise ValueError("window_len must be >= 1")
    st = max(1, int(step))

    out_dir_p = Path(out_dir).resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(cand_path))
    try:
        N = int(doc.page_count)
        rows: list[dict[str, Any]] = []
        if N >= K:
            for start in range(0, N - K + 1, st):
                end = int(start + K - 1)
                out_path = (out_dir_p / f"win_p{start:04d}-{end:04d}.pdf").resolve()
                if overwrite or (not out_path.exists()):
                    extract_window_pdf_file(src_doc=doc, start_page=int(start), end_page=int(end), out_path=out_path)

                # Hash by raw file bytes only (no PDF parsing).
                fp = fingerprint_file(out_path)
                rows.append(
                    {
                        "start_page": int(start),
                        "end_page": int(end),
                        "pdf": str(out_path),
                        "fingerprint": {
                            "size_bytes": int(fp.size_bytes),
                            "sha256": str(fp.sha256),
                            "tlsh": fp.tlsh,
                        },
                    }
                )

        return {
            "candidate": {
                "pdf": str(cand_path),
                "page_count": int(N),
            },
            "window_len": int(K),
            "step": int(st),
            "windows": rows,
        }
    finally:
        try:
            doc.close()
        except Exception:
            pass


def pick_best_window_by_tlsh(
    *,
    windows: list[dict[str, Any]],
    model_tlsh: str | None,
    max_dist: int = 200,
) -> dict[str, Any] | None:
    """
    Pick best window from a list produced by `extract_windows_and_fingerprint`.
    """
    if not model_tlsh or not tlsh_available():
        return None

    best_sim = -1.0
    best = None
    for w in windows:
        fp = (w.get("fingerprint") or {}) if isinstance(w, dict) else {}
        wt = fp.get("tlsh")
        sim = tlsh_similarity(wt, model_tlsh, max_dist=int(max_dist))
        dist = tlsh_distance(wt, model_tlsh)
        if sim is None or dist is None:
            continue
        if float(sim) > best_sim:
            best_sim = float(sim)
            best = {
                "start_page": int(w.get("start_page") or 0),
                "end_page": int(w.get("end_page") or 0),
                "window_pdf": str(w.get("pdf") or ""),
                "similarity": float(sim),
                "distance": int(dist),
                "candidate_tlsh": wt,
                "candidate_sha256": fp.get("sha256"),
                "candidate_size_bytes": fp.get("size_bytes"),
            }
    return best
