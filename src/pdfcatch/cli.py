from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import unicodedata
from importlib import metadata as importlib_metadata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Callable

from dotenv import load_dotenv

import fitz  # PyMuPDF

from pdfcatch.main import TemplateRef
from Modules.detector.image.dhash import dhash_score, page_dhash
from Modules.detector.stream.contents import fingerprint_page_contents, open_pdf as open_pikepdf, stream_similarity
from Modules.detector.text.metrics import score_text
from Modules.extrator.text_anchors import extract_pdf_by_anchors, load_anchor_rules

try:
    import tlsh as _tlsh  # optional (py-tlsh)
except Exception:
    _tlsh = None

try:
    import numpy as _np  # optional (kmeans mode)
except Exception:
    _np = None

try:
    from sklearn.cluster import KMeans as _SkKMeans  # optional (kmeans mode)
except Exception:
    _SkKMeans = None

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


_SAVE_KW = dict(
    garbage=4,
    clean=True,
    deflate=True,
    no_new_id=True,
    preserve_metadata=0,
)

_CMD_BLUE = "#4A90E2"
_DIM = "#7f8c8d"
_APP_NAME = "pdfcatch"
_ALL_METHODS = ["bytes", "tlsh", "stream", "text", "image"]
_METHOD_COLORS = {
    "bytes": "#5f6f7f",
    "tlsh": "#6a6f7a",
    "stream": "#5f7a6f",
    "text": "#b87838",
    "image": "#75657a",
}
_KMEANS_FEATURE_FIELDS = ["log_p1", "log_p2", "log_total", "ratio_p2_p1", "spread_max_min"]


def _app_version() -> str:
    try:
        return str(importlib_metadata.version(_APP_NAME))
    except Exception:
        return "0.0.0"


def _print_cli_help(console: Console, *, topic: str | None = None) -> None:
    topic_norm = str(topic or "").strip().lower()
    header = (
        f"[bold {_CMD_BLUE}]pdfcatch CLI[/bold {_CMD_BLUE}]  "
        f"[{_DIM}]v{_app_version()}[/]"
    )
    if topic_norm:
        header = header + f"\n[{_DIM}]Ajuda: {topic_norm}[/]"
    console.print(Panel.fit(header, border_style=_CMD_BLUE))

    cmds = Table(box=None, show_header=True, header_style=f"bold {_DIM}", pad_edge=False, expand=True)
    cmds.add_column("Comando", style=f"bold {_CMD_BLUE}", no_wrap=True)
    cmds.add_column("Descricao", style="white")
    cmds.add_row("doc", "Deteccao/extracao por label (DESPACHO ou CERTIDAO_CM) via metodos de similaridade.")
    cmds.add_row("extract-text", "Extracao textual por ancoras before/after (sem ROI/OCR/imagem).")
    cmds.add_row("studio", "Abre a interface integrada (Deteccao + Kit ROI + Extracao + Estatisticas) com otimizacao global vetorial de ancoras.")
    cmds.add_row("web", "Abre a WebUI focada em pasta/fluxo (modo tecnico).")
    cmds.add_row("tui", "Abre a TUI (terminal) para acompanhamento por pagina.")
    console.print(cmds)

    if topic_norm in {"doc", "document"}:
        console.print("\n[bold]Uso rapido: doc[/bold]")
        console.print(f"[{_CMD_BLUE}]cli doc -d :Q4-10 -m all --save-files outputs --web[/]")
        console.print(f"[{_CMD_BLUE}]cli doc -c :Q1-20 -m bytes --top-n 10[/]")
        console.print(f"[{_CMD_BLUE}]cli doc -d :Q8-8 -m bytes --top-n 1 --return[/]")
    elif topic_norm in {"extract-text", "alt-extract", "extract"}:
        console.print("\n[bold]Uso rapido: extract-text[/bold]")
        console.print(f"[{_CMD_BLUE}]cli extract-text --pdf :Q8-8 --anchors configs/anchor_text_fields.example.json[/]")
        console.print(f"[{_CMD_BLUE}]cli alt-extract --pdf :Q1-10 --anchors configs/anchor_text_fields.example.json --out outputs/anchor_text.json[/]")
        console.print(f"[{_CMD_BLUE}]cli extract-text --from-return latest --anchors configs/anchor_text_fields.example.json[/]")
    elif topic_norm in {"studio", "kit", "roi"}:
        console.print("\n[bold]Uso rapido: studio[/bold]")
        console.print(f"[{_CMD_BLUE}]cli studio --pdfs :Q4-10[/]")
        console.print(f"[{_CMD_BLUE}]cli studio --dir /mnt/c/Users/pichau/Desktop/geral_pdf/quarentena[/]")
        console.print(f"[{_CMD_BLUE}]cli studio --pdfs :Q4-10 --save-files outputs[/]")
    elif topic_norm in {"web", "ui"}:
        console.print("\n[bold]Uso rapido: web[/bold]")
        console.print(f"[{_CMD_BLUE}]cli web --pdfs :Q --section Deteccao[/]")
        console.print(f"[{_CMD_BLUE}]cli web --dir ./PDFs --section Estatisticas[/]")
    elif topic_norm in {"tui"}:
        console.print("\n[bold]Uso rapido: tui[/bold]")
        console.print(f"[{_CMD_BLUE}]cli tui /caminho/candidato.pdf --templates configs/templates.json[/]")
        console.print(f"[{_CMD_BLUE}]cli tui[/]")
    else:
        console.print("\n[bold]Exemplos[/bold]")
        examples = textwrap.dedent(
            f"""
            [{_CMD_BLUE}]cli studio --pdfs :Q4-10[/]
            [{_CMD_BLUE}]cli doc -d :Q4-10 -m all --save-files outputs --web[/]
            [{_CMD_BLUE}]cli doc -c :Q3-8 -m bytes --top-n 5[/]
            [{_CMD_BLUE}]cli tui[/]
            [{_CMD_BLUE}]cli help studio[/]
            """
        ).strip()
        console.print(examples)
    console.print(f"\n[{_DIM}]Dica:[/] use [bold {_CMD_BLUE}]cli <comando> --help[/] para parametros detalhados.")


def _find_repo_root() -> Path:
    """
    Best-effort find the repo root when running from within this repository.
    We avoid requiring flags for env/config paths.
    """
    cwd = Path.cwd().resolve()
    for p in [cwd] + list(cwd.parents):
        if (p / "configs" / "run.env").exists():
            return p
        if (p / "pyproject.toml").exists() and (p / "src" / "pdfcatch").exists():
            return p
    return cwd


def _safe_stem(s: str, *, max_len: int = 80) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._-")
    return (out[:max_len] or "pdf")


def _progress_bar(done: int, total: int, *, width: int = 24) -> str:
    total_i = int(max(0, total))
    done_i = int(max(0, done))
    if total_i <= 0:
        return f"|{'.' * int(max(1, width))}| 0/0 (0.00%)"
    done_i = min(done_i, total_i)
    pct = (float(done_i) / float(total_i)) * 100.0
    filled = int(round((float(done_i) / float(total_i)) * float(width)))
    filled = max(0, min(int(width), filled))
    bar = ("#" * filled) + ("." * (int(width) - filled))
    return f"|{bar}| {done_i}/{total_i} ({pct:0.2f}%)"


def _process_id_from_pdf_name(path: Path) -> str:
    """
    Process identifier used in logs/UI.
    Preference:
      1) first long numeric token in filename stem
      2) full stem as fallback
    """
    stem = path.stem
    m = re.search(r"(\d{6,})", stem)
    if m:
        return str(m.group(1))
    return stem


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _env_float(name: str, default: float) -> float:
    try:
        raw = (os.getenv(name) or "").strip()
        if not raw:
            return float(default)
        return float(raw)
    except Exception:
        return float(default)


def _resolve(root: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (root / pp).resolve()


def _infer_primary_pdf_dir(root: Path, specs: list[str]) -> Path | None:
    """
    Infer the most relevant input directory from user specs.
    Priority:
      1) explicit directory in spec metadata
      2) parent dir of the first expanded PDF
    """
    for spec in specs:
        try:
            expanded, meta = _expand_spec_to_pdfs(root, spec)
        except Exception:
            continue
        mdir = meta.get("dir")
        if isinstance(mdir, str) and mdir.strip():
            p = _resolve(root, mdir.strip())
            if p.exists() and p.is_dir():
                return p
        if expanded:
            p0 = expanded[0].resolve().parent
            if p0.exists() and p0.is_dir():
                return p0
    return None


def _launch_webui(
    *,
    root: Path,
    active_pdf_dir: Path | None,
    extracted_dir: Path | None,
    section: str = "Extracao",
    limit: int | None = None,
) -> int:
    env = os.environ.copy()
    if active_pdf_dir is not None:
        env["PDFCATCH_WEB_ACTIVE_PDF_DIR"] = str(active_pdf_dir.resolve())
        # Pin source dir explicitly to avoid inheriting alias/range from external shell env.
        env["PDFCATCH_PDF_DIR"] = str(active_pdf_dir.resolve())
    if extracted_dir is not None:
        env["PDFCATCH_WEB_EXTRACTED_DIR"] = str(extracted_dir.resolve())
    env["PDFCATCH_WEB_SECTION"] = str(section)
    env["PDFCATCH_WEB_SHOW_THUMBS"] = "1"
    env["PDFCATCH_WEB_SHOW_GALLERY"] = "1"
    # Web UI should not start capped at 10 unless user asks for a cap.
    env["PDFCATCH_LIMIT"] = str(int(limit) if limit is not None else 0)

    web_entry = (root / "Modules" / "web" / "webui_run.py").resolve()
    cmd = [sys.executable, str(web_entry)]
    return int(subprocess.call(cmd, cwd=str(root), env=env))


def _launch_tui(
    *,
    root: Path,
    argv: list[str],
) -> int:
    tui_entry = (root / "Modules" / "web" / "tui.py").resolve()
    cmd = [sys.executable, str(tui_entry)] + list(argv)
    return int(subprocess.call(cmd, cwd=str(root), env=os.environ.copy()))


def _parse_range_1_based(spec: str) -> tuple[int | None, int | None]:
    """
    Parse a 1-based inclusive range:
      "10-40" => (10, 40)
      "10"    => (10, 10)
      "-40"   => (None, 40)
      "10-"   => (10, None)
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


def _list_pdfs(dir_path: Path) -> list[Path]:
    items = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    items.sort(key=lambda p: p.name.lower())
    return items


def _expand_spec_to_pdfs(root: Path, spec: str) -> tuple[list[Path], dict[str, Any]]:
    """
    Expand a spec into concrete PDF paths.

    Spec formats:
      :Q4-10    => alias Q => $PDFCATCH_ALIAS_Q, range 1-based inclusive over PDFs in that dir
      /abs/dir  => take all PDFs in directory
      /abs/x.pdf => single PDF file
      relative/dir or relative/x.pdf => resolved against repo root
    """
    s = (spec or "").strip()
    if not s:
        raise ValueError("empty spec")

    meta: dict[str, Any] = {"spec": s}

    if s.startswith(":"):
        m = re.match(r"^:([A-Za-z])(.*)$", s)
        if not m:
            raise ValueError(f"invalid alias spec: {s!r}")
        alias = str(m.group(1)).upper()
        rest = (m.group(2) or "").strip()
        alias_path_raw = (os.getenv(f"PDFCATCH_ALIAS_{alias}") or "").strip()
        if not alias_path_raw:
            raise ValueError(f"missing env PDFCATCH_ALIAS_{alias} (required for {s!r})")
        base_dir = _resolve(root, alias_path_raw)
        if not base_dir.exists() or not base_dir.is_dir():
            raise ValueError(f"alias dir does not exist: {alias} => {base_dir}")
        start_1, end_1 = _parse_range_1_based(rest) if rest else (None, None)
        pdfs_all = _list_pdfs(base_dir)
        meta |= {"alias": alias, "dir": str(base_dir), "total": len(pdfs_all), "range": rest or None}
        if start_1 is not None or end_1 is not None:
            total = len(pdfs_all)
            s0 = (int(start_1) - 1) if start_1 is not None else 0
            e0 = int(end_1) if end_1 is not None else total
            s0 = max(0, min(s0, total))
            e0 = max(0, min(e0, total))
            return (pdfs_all[s0:e0], meta | {"selected": e0 - s0, "start_1": start_1, "end_1": end_1})
        return (pdfs_all, meta | {"selected": len(pdfs_all), "start_1": None, "end_1": None})

    p = _resolve(root, s)
    if p.exists() and p.is_file() and p.suffix.lower() == ".pdf":
        return ([p], meta | {"path": str(p), "selected": 1})
    if p.exists() and p.is_dir():
        pdfs = _list_pdfs(p)
        return (pdfs, meta | {"dir": str(p), "total": len(pdfs), "selected": len(pdfs)})

    raise ValueError(f"spec path not found: {s} => {p}")


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
        if not pdf_raw:
            raise ValueError(f"template[{i}]: missing 'pdf'")
        pdf_path = _resolve(root, str(pdf_raw))
        page_index = int(it.get("page_index") or 0)
        out.append(TemplateRef(id=tid, label=label, pdf_path=str(pdf_path), page_index=page_index))
    return out


def _size_similarity(a: int, b: int) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return float(min(a, b) / max(a, b))


def _one_page_pdf_bytes(doc: fitz.Document, *, page_index: int) -> bytes:
    out = fitz.open()
    try:
        out.insert_pdf(doc, from_page=int(page_index), to_page=int(page_index))
        return out.tobytes(**_SAVE_KW)
    finally:
        try:
            out.close()
        except Exception:
            pass


def _window_pdf_bytes(doc: fitz.Document, *, start_page: int, end_page: int) -> bytes:
    out = fitz.open()
    try:
        # widgets=False avoids form-widget copy issues on some PDFs when extracting windows.
        out.insert_pdf(doc, from_page=int(start_page), to_page=int(end_page), widgets=False)
        return out.tobytes(**_SAVE_KW)
    finally:
        try:
            out.close()
        except Exception:
            pass


def _pages_pdf_bytes(doc: fitz.Document, *, page_indices: list[int]) -> bytes:
    out = fitz.open()
    try:
        for pi in page_indices:
            out.insert_pdf(doc, from_page=int(pi), to_page=int(pi), widgets=False)
        return out.tobytes(**_SAVE_KW)
    finally:
        try:
            out.close()
        except Exception:
            pass


def _export_window_pdf(src_doc: fitz.Document, *, start_page: int, end_page: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = fitz.open()
    try:
        out.insert_pdf(src_doc, from_page=int(start_page), to_page=int(end_page), widgets=False)
        data = out.tobytes(**_SAVE_KW)
        out_path.write_bytes(data)
    finally:
        try:
            out.close()
        except Exception:
            pass


def _export_pages_pdf(src_doc: fitz.Document, *, page_indices: list[int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = fitz.open()
    try:
        for pi in page_indices:
            out.insert_pdf(src_doc, from_page=int(pi), to_page=int(pi), widgets=False)
        data = out.tobytes(**_SAVE_KW)
        out_path.write_bytes(data)
    finally:
        try:
            out.close()
        except Exception:
            pass


@dataclass
class BestWindow:
    start_page: int
    end_page: int
    total: float
    per_page: list[dict[str, Any]]
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class SlotSizeModel:
    slot: int
    template_page_index: int
    model_size: int
    median_size: float
    mad_size: float
    sample_count: int


def _page_indices_from_window(window: Any) -> list[int]:
    if not isinstance(window, dict):
        return []
    pidx = window.get("page_indices")
    if isinstance(pidx, list):
        out: list[int] = []
        for v in pidx:
            if isinstance(v, int):
                out.append(int(v))
        if out:
            return out
    sidx = window.get("start_page_index")
    eidx = window.get("end_page_index")
    if isinstance(sidx, int) and isinstance(eidx, int):
        if int(sidx) <= int(eidx):
            return list(range(int(sidx), int(eidx) + 1))
        return [int(sidx), int(eidx)]
    return []


def _idx0_compact(page_indices: list[int]) -> str:
    if not page_indices:
        return "-"
    if len(page_indices) == 1:
        return f"{int(page_indices[0]):04d}"
    if len(page_indices) == 2:
        a = int(page_indices[0])
        b = int(page_indices[1])
        if b == a + 1:
            return f"{a:04d}-{b:04d}"
        return f"{a:04d}+{b:04d}"
    return "+".join(f"{int(pi):04d}" for pi in page_indices)


def _pages_compact(page_indices: list[int]) -> str:
    if not page_indices:
        return "-"
    one_based = [int(pi) + 1 for pi in page_indices]
    if len(one_based) == 1:
        return str(one_based[0])
    if len(one_based) == 2:
        a = int(one_based[0])
        b = int(one_based[1])
        if b == a + 1:
            return f"{a}-{b}"
        return f"{a}+{b}"
    return "+".join(str(v) for v in one_based)


def _window_to_idx_pages(window: Any) -> tuple[str, str]:
    page_indices = _page_indices_from_window(window)
    if page_indices:
        return (_idx0_compact(page_indices), _pages_compact(page_indices))
    return ("-", "-")


def _method_pages_line(ps: dict[str, Any], *, colored: bool = False) -> str:
    methods_map = ps.get("methods") if isinstance(ps.get("methods"), dict) else {}
    selected = ps.get("selected_methods")
    methods: list[str]
    if isinstance(selected, list):
        methods = [str(m).strip().lower() for m in selected if str(m).strip()]
    else:
        methods = [str(m).strip().lower() for m in methods_map.keys() if str(m).strip()]
        methods = [m for m in _ALL_METHODS if m in methods] + [m for m in methods if m not in _ALL_METHODS]
    if not methods:
        return "-"

    parts: list[str] = []
    for m in methods:
        md = methods_map.get(m) if isinstance(methods_map, dict) else None
        bw = md.get("best_window") if isinstance(md, dict) else None
        idx0, pages = _window_to_idx_pages(bw)
        value = "-" if idx0 == "-" else f"idx{idx0}/p{pages}"
        if colored:
            c = _METHOD_COLORS.get(m, "white")
            parts.append(f"[{c}]{m}[/{c}]={value}")
        else:
            parts.append(f"{m}={value}")
    return ", ".join(parts) if parts else "-"


def _method_short_name(method: str) -> str:
    m = str(method or "").strip().lower()
    return {
        "bytes": "BYT",
        "tlsh": "TLS",
        "stream": "STR",
        "text": "TXT",
        "image": "IMG",
    }.get(m, m[:3].upper() if m else "MTH")


def _report_methods(rows: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for ps in rows:
        selected = ps.get("selected_methods")
        if isinstance(selected, list):
            for m in selected:
                key = str(m).strip().lower()
                if key and key not in seen:
                    seen.append(key)
    if not seen:
        for ps in rows:
            methods_map = ps.get("methods")
            if isinstance(methods_map, dict):
                for m in methods_map.keys():
                    key = str(m).strip().lower()
                    if key and key not in seen:
                        seen.append(key)
    return [m for m in _ALL_METHODS if m in seen] + [m for m in seen if m not in _ALL_METHODS]


def _method_cell_compact(ps: dict[str, Any], method: str, *, colored: bool = False) -> str:
    methods_map = ps.get("methods")
    if not isinstance(methods_map, dict):
        return "-"
    md = methods_map.get(method)
    if not isinstance(md, dict):
        return "-"
    if str(md.get("status") or "") != "ok":
        return "-"
    bw = md.get("best_window")
    _idx, pages = _window_to_idx_pages(bw)
    score = md.get("score")
    if pages == "-" or not isinstance(score, (int, float)):
        return "-"
    text = f"p{pages}|{float(score):.3f}"
    if not colored:
        return text
    c = _METHOD_COLORS.get(str(method).lower(), "white")
    return f"[{c}]{text}[/{c}]"


def _method_detail_tuple(ps: dict[str, Any], method: str) -> tuple[str, str, float | None]:
    methods_map = ps.get("methods")
    if not isinstance(methods_map, dict):
        return ("-", "-", None)
    md = methods_map.get(method)
    if not isinstance(md, dict):
        return ("-", "-", None)
    if str(md.get("status") or "") != "ok":
        return ("-", "-", None)
    bw = md.get("best_window")
    idx0, pages = _window_to_idx_pages(bw)
    score = md.get("score")
    sc = float(score) if isinstance(score, (int, float)) else None
    return (idx0, pages, sc)


class LiveRunDashboard:
    """
    Live dashboard without log-region redraw:
      - top: summary table (updated only when summary changes)
      - processing lines are printed naturally below
    """

    def __init__(self, *, console: Console, expected_rows: int = 0, max_log_lines: int = 300) -> None:
        self.console = console
        self.expected_rows = int(max(0, expected_rows))
        self._summary_rows: list[dict[str, Any]] = []
        self._live: Live | None = None

    def start(self) -> None:
        if self._live is not None:
            return
        self._live = Live(
            self._summary_table(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
            auto_refresh=False,
            vertical_overflow="visible",
        )
        self._live.start()

    def stop(self) -> None:
        if self._live is None:
            return
        self._live.stop()
        self._live = None

    def log(self, text: str = "", style: str | None = None) -> None:
        if self._live is not None:
            if style:
                self._live.console.print(text, style=style)
            else:
                self._live.console.print(text)
            return
        if style:
            self.console.print(text, style=style)
        else:
            self.console.print(text)

    def set_summaries(self, rows: list[dict[str, Any]]) -> None:
        self._summary_rows = list(rows)
        self.refresh()

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._summary_table(), refresh=True)

    def _summary_table(self) -> Table:
        rows = self._summary_rows if self._summary_rows else []
        table = Table(
            box=None,
            expand=True,
            header_style="bold bright_black",
            show_lines=False,
            pad_edge=False,
        )
        table.add_column("#", justify="right", no_wrap=True)
        table.add_column("Arquivo", justify="left", no_wrap=True, overflow="ellipsis")
        table.add_column("Label", justify="center", no_wrap=True)
        table.add_column("Final", justify="right", no_wrap=True)
        table.add_column("Idx0", justify="center", no_wrap=True)

        start_idx = 0
        for pos, ps in enumerate(rows, start=1):
            ridx = start_idx + pos
            cand = Path(str(ps.get("candidate_pdf") or "-")).name
            lab = str(ps.get("label") or "-")
            fs = ps.get("final_score")
            final_txt = f"{float(fs):.4f} ({float(fs)*100.0:0.2f}%)" if isinstance(fs, (int, float)) else "-"
            win_idx = str(ps.get("best_idx0") or "-")
            table.add_row(str(ridx), cand, lab, final_txt, win_idx)

        if not rows:
            table.add_row("-", "-", "-", "-", "-")
        return table


def _print_consolidated_report(console: Console, process_summaries: list[dict[str, Any]]) -> None:
    if not process_summaries:
        return
    methods = _report_methods(process_summaries)
    console.print("Relatorio final consolidado", style="bold white")
    summary_table = Table(
        box=None,
        expand=True,
        header_style="bold bright_black",
        show_lines=False,
        pad_edge=True,
        row_styles=["none", "dim"],
    )
    summary_table.add_column("#", justify="right", no_wrap=True)
    summary_table.add_column("Arquivo", justify="left", no_wrap=True, overflow="ellipsis")
    summary_table.add_column("Label", justify="center", no_wrap=True)
    summary_table.add_column("Final", justify="right", no_wrap=True)
    summary_table.add_column("Idx0", justify="center", no_wrap=True)

    for ridx, ps in enumerate(process_summaries, start=1):
        cand = Path(str(ps.get("candidate_pdf") or "-")).name
        lab = str(ps.get("label") or "-")
        fs = ps.get("final_score")
        final_txt = f"{float(fs):.4f} ({float(fs)*100.0:0.2f}%)" if isinstance(fs, (int, float)) else "-"
        win_idx = str(ps.get("best_idx0") or "-")
        summary_table.add_row(str(ridx), cand, lab, final_txt, win_idx)

    console.print(summary_table)
    if methods:
        console.print("Detalhe por metodo", style="bold white")
        for ridx, ps in enumerate(process_summaries, start=1):
            cand = Path(str(ps.get("candidate_pdf") or "-")).name
            fs = ps.get("final_score")
            final_txt = f"{float(fs):.4f}" if isinstance(fs, (int, float)) else "-"
            win_idx = str(ps.get("best_idx0") or "-")
            console.print(f"{ridx:>2}. {cand} | final={final_txt} | idx0={win_idx}", style="bright_black")
            for m in methods:
                idx0, pages, sc = _method_detail_tuple(ps, m)
                c = _METHOD_COLORS.get(str(m).lower(), "white")
                if idx0 == "-" or sc is None:
                    console.print(f"    [{c}]{_method_short_name(m)}[/{c}] -> -")
                else:
                    console.print(f"    [{c}]{_method_short_name(m)}[/{c}] -> idx{idx0}/p{pages} | {sc:.4f}")


def _rank_windows_by_size(
    *,
    candidate_pdf: Path,
    model_page_sizes: list[int],
    min_p1_override: float | None = None,
    min_p2_override: float | None = None,
    kmeans_ctx: dict[str, Any] | None = None,
) -> tuple[list[BestWindow], int, int]:
    K = int(len(model_page_sizes))
    if K <= 0:
        return ([], 0, 0)

    cand = fitz.open(str(candidate_pdf))
    try:
        N = int(cand.page_count)
        if N < K:
            return ([], N, 0)

        # Candidate page sizes (1-page mini-PDF length), opaque bytes only.
        cand_sizes = [int(len(_one_page_pdf_bytes(cand, page_index=i))) for i in range(N)]

        ranked_ok: list[BestWindow] = []
        ranked_all: list[BestWindow] = []
        min_any = float(_env_float("PDFCATCH_BYTES_MIN_ANY", 0.40))
        min_p1 = float(min_p1_override) if min_p1_override is not None else float(_env_float("PDFCATCH_BYTES_MIN_P1", 0.60))
        min_p2 = float(min_p2_override) if min_p2_override is not None else float(_env_float("PDFCATCH_BYTES_MIN_P2", 0.40))
        kmeans_enabled = isinstance(kmeans_ctx, dict) and bool(kmeans_ctx)
        centroids = kmeans_ctx.get("centroids") if kmeans_enabled else None
        target_cluster = int(kmeans_ctx.get("target_cluster")) if kmeans_enabled and isinstance(kmeans_ctx.get("target_cluster"), int) else 0
        target_centroid = kmeans_ctx.get("target_centroid") if kmeans_enabled else None
        weights = kmeans_ctx.get("weights") if kmeans_enabled else None
        w_size = float(weights.get("size") if isinstance(weights, dict) else 0.70) if kmeans_enabled else 1.0
        w_cluster = float(weights.get("cluster") if isinstance(weights, dict) else 0.30) if kmeans_enabled else 0.0
        if kmeans_enabled:
            norm = float(w_size + w_cluster)
            if norm <= 0.0:
                w_size, w_cluster = 1.0, 0.0
            else:
                w_size = float(w_size / norm)
                w_cluster = float(w_cluster / norm)

        for start in range(0, N - K + 1):
            per: list[dict[str, Any]] = []
            scores: list[float] = []
            size_bytes_window: list[int] = []
            for pos in range(K):
                pi = int(start + pos)
                csz = int(cand_sizes[pi])
                size_bytes_window.append(int(csz))
                msz = int(model_page_sizes[pos])
                sc = float(_size_similarity(msz, csz))
                per.append({"page_index": pi, "size_bytes": csz, "score": sc})
                scores.append(sc)
            size_total = float(sum(scores) / float(len(scores) or 1))

            total = float(size_total)
            meta: dict[str, Any] = {"size_total": float(size_total)}
            if kmeans_enabled and isinstance(target_centroid, list):
                feat = _window_feature_from_sizes(size_bytes_window)
                d_target = _euclidean_distance(feat, [float(v) for v in target_centroid])
                sim_target = float(1.0 / (1.0 + d_target))
                nearest_cluster = target_cluster
                nearest_dist = d_target
                if isinstance(centroids, list) and centroids:
                    all_d = [_euclidean_distance(feat, [float(v) for v in c]) for c in centroids if isinstance(c, list)]
                    if all_d:
                        nearest_cluster = int(min(range(len(all_d)), key=lambda i: float(all_d[i])))
                        nearest_dist = float(all_d[nearest_cluster])
                total = float((w_size * size_total) + (w_cluster * sim_target))
                meta.update(
                    {
                        "kmeans_feature": [float(v) for v in feat],
                        "kmeans_similarity": float(sim_target),
                        "kmeans_distance": float(d_target),
                        "nearest_cluster": int(nearest_cluster),
                        "nearest_cluster_distance": float(nearest_dist),
                        "target_cluster": int(target_cluster),
                        "is_target_cluster": bool(int(nearest_cluster) == int(target_cluster)),
                        "final_total": float(total),
                    }
                )
            w = BestWindow(start_page=int(start), end_page=int(start + K - 1), total=total, per_page=per, meta=meta)
            ranked_all.append(w)

            # Eligibility for K=2 (DESPACHO):
            # - no page below MIN_ANY
            # - page1 > MIN_P1
            # - page2 >= MIN_P2
            eligible = True
            if K == 2 and len(scores) >= 2:
                s1 = float(scores[0])
                s2 = float(scores[1])
                if (s1 < min_any) or (s2 < min_any):
                    eligible = False
                if not (s1 > min_p1):
                    eligible = False
                if not (s2 >= min_p2):
                    eligible = False
            if eligible:
                ranked_ok.append(w)

        # Strict eligibility: return only windows that satisfy thresholds.
        ranked_ok.sort(
            key=lambda w: (
                float(w.total),
                float((w.meta or {}).get("size_total") or 0.0),
            ),
            reverse=True,
        )
        return (ranked_ok, N, max(0, N - K + 1))
    finally:
        try:
            cand.close()
        except Exception:
            pass


def _rank_windows_by_mean2(
    *,
    candidate_pdf: Path,
    mean_page_sizes: list[float],
) -> tuple[list[BestWindow], int, int]:
    K = int(len(mean_page_sizes))
    if K <= 0:
        return ([], 0, 0)
    cand = fitz.open(str(candidate_pdf))
    try:
        N = int(cand.page_count)
        if N < K:
            return ([], N, 0)
        cand_sizes = [int(len(_one_page_pdf_bytes(cand, page_index=i))) for i in range(N)]
        ranked: list[BestWindow] = []
        for start in range(0, N - K + 1):
            per: list[dict[str, Any]] = []
            dists: list[float] = []
            for pos in range(K):
                pi = int(start + pos)
                csz = int(cand_sizes[pi])
                mean_sz = float(mean_page_sizes[pos])
                denom = max(1.0, float(abs(mean_sz)))
                dist = float(abs(float(csz) - float(mean_sz)) / denom)
                slot_score = float(1.0 / (1.0 + dist))
                per.append(
                    {
                        "page_index": pi,
                        "size_bytes": csz,
                        "mean_size": float(mean_sz),
                        "distance_norm": float(dist),
                        "score": float(slot_score),
                    }
                )
                dists.append(float(dist))
            avg_dist = float(sum(dists) / float(len(dists) or 1))
            total = float(1.0 / (1.0 + avg_dist))
            ranked.append(
                BestWindow(
                    start_page=int(start),
                    end_page=int(start + K - 1),
                    total=float(total),
                    per_page=per,
                    meta={"mean_distance": float(avg_dist), "selection_rule": "nearest_mean_distance"},
                )
            )
        ranked.sort(
            key=lambda w: (
                float(w.total),
                -float((w.meta or {}).get("mean_distance") or 0.0),
            ),
            reverse=True,
        )
        return (ranked, N, max(0, N - K + 1))
    finally:
        try:
            cand.close()
        except Exception:
            pass


def _group_templates_by_slot(templates_seq: list[TemplateRef]) -> list[list[TemplateRef]]:
    grouped: dict[int, list[TemplateRef]] = {}
    for t in templates_seq:
        pi = int(t.page_index)
        grouped.setdefault(pi, []).append(t)

    out: list[list[TemplateRef]] = []
    for pi in sorted(grouped.keys()):
        grp = list(grouped.get(pi) or [])
        grp.sort(key=lambda t: (Path(str(t.pdf_path)).name.lower(), str(t.id).lower()))
        if grp:
            out.append(grp)
    return out


def _slot_representatives(slot_templates: list[list[TemplateRef]]) -> list[TemplateRef]:
    reps: list[TemplateRef] = []
    for grp in slot_templates:
        if grp:
            reps.append(grp[0])
    return reps


def _median_float(values: list[int]) -> float:
    if not values:
        return 0.0
    return float(median([int(v) for v in values]))


def _mad_float(values: list[int], *, center: float) -> float:
    if not values:
        return 0.0
    dev = [abs(float(int(v)) - float(center)) for v in values]
    return float(median(dev))


def _fold_for_match(text: str) -> str:
    out: list[str] = []
    for ch in (text or ""):
        decomp = unicodedata.normalize("NFKD", ch)
        for dc in decomp:
            if unicodedata.combining(dc):
                continue
            out.append(dc.casefold())
    return "".join(out)


def _split_env_terms(raw: str) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    for part in str(raw).replace(",", "|").split("|"):
        t = str(part).strip()
        if t:
            out.append(t)
    return out


def _find_kmeans_anchor_source(root: Path) -> Path | None:
    explicit = str(os.getenv("PDFCATCH_KMEANS_ANCHORS_JSON") or "").strip()
    if explicit:
        p = _resolve(root, explicit)
        if p.exists() and p.is_file():
            return p
        return None

    cand_paths = [
        (root / ".." / "pdfcatch-extractor" / "configs" / "anchor_text_fields.json").resolve(),
        (root / "configs" / "anchor_text_fields.json").resolve(),
        (root / "configs" / "anchor_text_fields.example.json").resolve(),
    ]
    for p in cand_paths:
        if p.exists() and p.is_file():
            return p
    return None


def _iter_anchor_terms_from_json(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("rules") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        for k in ["before", "after"]:
            v = it.get(k)
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
        for k in ["before_any", "after_any", "before_variants", "after_variants"]:
            vv = it.get(k)
            if isinstance(vv, list):
                for x in vv:
                    if isinstance(x, str) and x.strip():
                        out.append(x.strip())
            elif isinstance(vv, str) and vv.strip():
                out.append(vv.strip())
    return out


def _load_kmeans_text_terms(root: Path) -> dict[str, Any]:
    req_env = _split_env_terms(str(os.getenv("PDFCATCH_KMEANS_REQUIRED_TERMS") or ""))
    required_terms = req_env if req_env else ["Diretoria Especial", "Requerente", "Interessad"]

    anchor_path = _find_kmeans_anchor_source(root)
    tip_terms: list[str] = []
    if anchor_path is not None:
        try:
            tip_terms.extend(_iter_anchor_terms_from_json(anchor_path))
        except Exception:
            tip_terms = []
    tip_terms.extend(_split_env_terms(str(os.getenv("PDFCATCH_KMEANS_EXTRA_TIPS") or "")))

    # Keep deterministic ordering while removing duplicates (case/accent-insensitive).
    seen: set[str] = set()
    req_final: list[str] = []
    for t in required_terms:
        tf = _fold_for_match(str(t)).strip()
        if not tf or tf in seen:
            continue
        seen.add(tf)
        req_final.append(str(t).strip())

    tip_seen: set[str] = set(seen)
    tips_final: list[str] = []
    for t in tip_terms:
        tt = str(t).strip()
        tf = _fold_for_match(tt).strip()
        if not tf or len(tf) < 3 or tf in tip_seen:
            continue
        tip_seen.add(tf)
        tips_final.append(tt)

    return {
        "required_terms": req_final,
        "tip_terms": tips_final,
        "anchor_source": str(anchor_path) if anchor_path is not None else None,
    }


def _window_feature_from_sizes(sizes: list[int]) -> list[float]:
    if len(sizes) < 2:
        raise ValueError("kmeans bytes feature requires at least 2 page sizes")
    p1 = max(1, int(sizes[0]))
    p2 = max(1, int(sizes[1]))
    total = max(1, int(sum(int(v) for v in sizes)))
    mn = max(1, int(min(sizes)))
    mx = max(1, int(max(sizes)))
    return [
        float(math.log1p(p1)),
        float(math.log1p(p2)),
        float(math.log1p(total)),
        float(float(p2) / float(p1)),
        float(float(mx) / float(mn)),
    ]


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    acc = 0.0
    for i in range(n):
        d = float(a[i]) - float(b[i])
        acc += (d * d)
    return float(math.sqrt(acc))


def _window_text_signal(
    *,
    folded_text: str,
    required_terms_fold: list[str],
    tip_terms_fold: list[str],
) -> dict[str, Any]:
    req_hits = 0
    for t in required_terms_fold:
        if t and t in folded_text:
            req_hits += 1
    tip_hits = 0
    for t in tip_terms_fold:
        if t and t in folded_text:
            tip_hits += 1
    req_total = max(1, len(required_terms_fold))
    tip_total = max(1, len(tip_terms_fold))
    return {
        "required_hits": int(req_hits),
        "required_total": int(len(required_terms_fold)),
        "required_rate": float(float(req_hits) / float(req_total)),
        "required_all": bool(req_hits >= len(required_terms_fold) and len(required_terms_fold) > 0),
        "tip_hits": int(tip_hits),
        "tip_total": int(len(tip_terms_fold)),
        "tip_rate": float(float(tip_hits) / float(tip_total)),
    }


def _build_kmeans_bytes_reference(
    *,
    root: Path,
    slot_templates: list[list[TemplateRef]],
    label: str,
    train_pdfs: list[Path],
) -> dict[str, Any]:
    if _np is None or _SkKMeans is None:
        raise RuntimeError("bytes_ref_mode=kmeans requires numpy + scikit-learn installed")
    K = int(len(slot_templates))
    if K != 2:
        raise ValueError(f"bytes_ref_mode=kmeans currently supports K=2 only (got K={K})")

    km_terms = _load_kmeans_text_terms(root)
    required_terms = [str(x) for x in (km_terms.get("required_terms") or []) if str(x).strip()]
    tip_terms = [str(x) for x in (km_terms.get("tip_terms") or []) if str(x).strip()]
    req_fold = [_fold_for_match(x) for x in required_terms]
    tip_fold = [_fold_for_match(x) for x in tip_terms]

    train_rows: list[dict[str, Any]] = []
    for pdf in train_pdfs:
        doc = fitz.open(str(pdf))
        try:
            N = int(doc.page_count)
            if N < K:
                continue
            page_sizes: list[int] = []
            page_text_fold: list[str] = []
            for pi in range(N):
                page_sizes.append(int(len(_one_page_pdf_bytes(doc, page_index=pi))))
                page_text_fold.append(_fold_for_match(doc.load_page(pi).get_text("text") or ""))

            for start in range(0, N - K + 1):
                sizes = [int(page_sizes[start + pos]) for pos in range(K)]
                feat = _window_feature_from_sizes(sizes)
                joined = " ".join(page_text_fold[start + pos] for pos in range(K))
                sig = _window_text_signal(folded_text=joined, required_terms_fold=req_fold, tip_terms_fold=tip_fold)
                train_rows.append(
                    {
                        "pdf": str(pdf),
                        "start_page_index": int(start),
                        "end_page_index": int(start + K - 1),
                        "sizes": sizes,
                        "feature": feat,
                        "signal": sig,
                    }
                )
        finally:
            try:
                doc.close()
            except Exception:
                pass

    if len(train_rows) < 2:
        raise RuntimeError(
            "bytes_ref_mode=kmeans needs at least 2 train windows; "
            f"got={len(train_rows)} (pdfs={len(train_pdfs)})"
        )

    n_clusters = int(os.getenv("PDFCATCH_KMEANS_CLUSTERS", "2") or "2")
    if n_clusters != 2:
        raise ValueError(f"PDFCATCH_KMEANS_CLUSTERS must be 2 (got {n_clusters})")

    X = _np.array([row["feature"] for row in train_rows], dtype=float)
    km = _SkKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = [int(v) for v in km.fit_predict(X).tolist()]
    centroids = [[float(x) for x in row] for row in km.cluster_centers_.tolist()]

    cluster_acc: dict[int, dict[str, Any]] = {}
    for row, cid in zip(train_rows, labels):
        acc = cluster_acc.setdefault(
            int(cid),
            {
                "cluster_id": int(cid),
                "count": 0,
                "required_all_count": 0,
                "required_rate_sum": 0.0,
                "tip_rate_sum": 0.0,
                "slot_sizes": [[] for _ in range(K)],
            },
        )
        acc["count"] = int(acc["count"]) + 1
        sig = row["signal"] if isinstance(row.get("signal"), dict) else {}
        if bool(sig.get("required_all")):
            acc["required_all_count"] = int(acc["required_all_count"]) + 1
        acc["required_rate_sum"] = float(acc["required_rate_sum"]) + float(sig.get("required_rate") or 0.0)
        acc["tip_rate_sum"] = float(acc["tip_rate_sum"]) + float(sig.get("tip_rate") or 0.0)
        sizes = row["sizes"] if isinstance(row.get("sizes"), list) else []
        for pos in range(K):
            if pos < len(sizes):
                acc["slot_sizes"][pos].append(int(sizes[pos]))

    cluster_summaries: list[dict[str, Any]] = []
    best_cluster: int | None = None
    best_key: tuple[float, float, float, int] | None = None
    for cid in sorted(cluster_acc.keys()):
        acc = cluster_acc[cid]
        count = max(1, int(acc["count"]))
        req_all_rate = float(float(acc["required_all_count"]) / float(count))
        req_mean_rate = float(float(acc["required_rate_sum"]) / float(count))
        tip_mean_rate = float(float(acc["tip_rate_sum"]) / float(count))
        rank_key = (req_all_rate, req_mean_rate, tip_mean_rate, int(acc["count"]))
        if best_key is None or rank_key > best_key:
            best_key = rank_key
            best_cluster = int(cid)
        cluster_summaries.append(
            {
                "cluster_id": int(cid),
                "count": int(acc["count"]),
                "required_all_rate": float(req_all_rate),
                "required_mean_rate": float(req_mean_rate),
                "tip_mean_rate": float(tip_mean_rate),
            }
        )

    if best_cluster is None:
        raise RuntimeError("kmeans failed to select target cluster")

    target_rows = [row for row, cid in zip(train_rows, labels) if int(cid) == int(best_cluster)]
    if not target_rows:
        raise RuntimeError("kmeans selected empty target cluster")

    slot_models: list[SlotSizeModel] = []
    for pos in range(K):
        rep = slot_templates[pos][0] if pos < len(slot_templates) and slot_templates[pos] else None
        sample_sizes = [int(row["sizes"][pos]) for row in target_rows if isinstance(row.get("sizes"), list) and pos < len(row["sizes"])]
        med = _median_float(sample_sizes)
        mad = _mad_float(sample_sizes, center=med)
        slot_models.append(
            SlotSizeModel(
                slot=int(pos + 1),
                template_page_index=int(rep.page_index) if rep is not None else int(pos),
                model_size=int(round(med)) if sample_sizes else 0,
                median_size=float(med),
                mad_size=float(mad),
                sample_count=int(len(sample_sizes)),
            )
        )

    w_size = float(_env_float("PDFCATCH_KMEANS_WEIGHT_SIZE", 0.70))
    w_cluster = float(_env_float("PDFCATCH_KMEANS_WEIGHT_CLUSTER", 0.30))
    if w_size < 0.0:
        w_size = 0.0
    if w_cluster < 0.0:
        w_cluster = 0.0
    norm = float(w_size + w_cluster)
    if norm <= 0.0:
        w_size, w_cluster = 1.0, 0.0
    else:
        w_size = float(w_size / norm)
        w_cluster = float(w_cluster / norm)

    return {
        "slot_size_models": slot_models,
        "kmeans_model": {
            "feature_fields": list(_KMEANS_FEATURE_FIELDS),
            "centroids": centroids,
            "target_cluster": int(best_cluster),
            "target_centroid": [float(v) for v in centroids[int(best_cluster)]],
            "weights": {"size": float(w_size), "cluster": float(w_cluster)},
        },
        "summary": {
            "label": str(label),
            "train_pdf_count": int(len(train_pdfs)),
            "train_window_count": int(len(train_rows)),
            "target_cluster_rule": "max(required_all_rate, required_mean_rate, tip_mean_rate, count)",
            "target_cluster": int(best_cluster),
            "cluster_summaries": cluster_summaries,
            "required_terms": required_terms,
            "tip_term_count": int(len(tip_terms)),
            "anchor_source": km_terms.get("anchor_source"),
            "weights": {"size": float(w_size), "cluster": float(w_cluster)},
            "feature_fields": list(_KMEANS_FEATURE_FIELDS),
            "centroids": centroids,
        },
    }


def _build_mean2_bytes_reference(
    *,
    slot_templates: list[list[TemplateRef]],
    label: str,
    train_pdfs: list[Path],
) -> dict[str, Any]:
    K = int(len(slot_templates))
    if K <= 0:
        raise ValueError(f"bytes_ref_mode=mean2 invalid slot count for label={label}")

    rows: list[list[int]] = []
    skipped_short = 0
    for pdf in train_pdfs:
        doc = fitz.open(str(pdf))
        try:
            N = int(doc.page_count)
            if N < K:
                skipped_short += 1
                continue
            # mean2 assumes treino ja segmentado em docs-alvo; usa janela inicial fixa.
            sz = [int(len(_one_page_pdf_bytes(doc, page_index=pos))) for pos in range(K)]
            rows.append(sz)
        finally:
            try:
                doc.close()
            except Exception:
                pass

    if len(rows) < 2:
        raise RuntimeError(
            f"bytes_ref_mode=mean2 needs >=2 train docs for label={label}; got={len(rows)}"
        )

    z_thr = float(_env_float("PDFCATCH_MEAN2_OUTLIER_Z", 3.0))
    pct_tol = float(_env_float("PDFCATCH_MEAN2_OUTLIER_PCT", 0.15))
    if z_thr <= 0:
        z_thr = 3.0
    if pct_tol < 0:
        pct_tol = 0.15

    slot_stats: list[dict[str, Any]] = []
    keep_masks: list[list[bool]] = []
    for pos in range(K):
        vals = [int(r[pos]) for r in rows]
        med = _median_float(vals)
        mad = _mad_float(vals, center=med)
        sigma = float(1.4826 * mad)
        mask: list[bool] = []
        if sigma > 0.0:
            lim = float(z_thr * sigma)
        else:
            lim = float(max(1.0, abs(med) * pct_tol))
        for v in vals:
            mask.append(bool(abs(float(v) - float(med)) <= float(lim)))
        keep_masks.append(mask)
        slot_stats.append(
            {
                "slot": int(pos + 1),
                "median_raw": float(med),
                "mad_raw": float(mad),
                "sigma_raw": float(sigma),
                "limit_abs": float(lim),
            }
        )

    kept_rows: list[list[int]] = []
    for ridx, r in enumerate(rows):
        ok = True
        for pos in range(K):
            if not keep_masks[pos][ridx]:
                ok = False
                break
        if ok:
            kept_rows.append(r)

    if len(kept_rows) < 2:
        kept_rows = list(rows)

    slot_size_models: list[SlotSizeModel] = []
    mean_sizes: list[float] = []
    std_sizes: list[float] = []
    for pos in range(K):
        vals = [int(r[pos]) for r in kept_rows]
        cnt = len(vals)
        m = float(sum(vals) / float(max(1, cnt)))
        var = float(sum((float(v) - m) ** 2 for v in vals) / float(max(1, cnt)))
        sd = float(math.sqrt(var))
        med = _median_float(vals)
        mad = _mad_float(vals, center=med)
        rep = slot_templates[pos][0] if pos < len(slot_templates) and slot_templates[pos] else None
        slot_size_models.append(
            SlotSizeModel(
                slot=int(pos + 1),
                template_page_index=int(rep.page_index) if rep is not None else int(pos),
                model_size=int(round(m)),
                median_size=float(med),
                mad_size=float(mad),
                sample_count=int(cnt),
            )
        )
        mean_sizes.append(float(m))
        std_sizes.append(float(sd))

    return {
        "slot_size_models": slot_size_models,
        "summary": {
            "label": str(label),
            "train_pdf_count": int(len(train_pdfs)),
            "train_doc_count_used_raw": int(len(rows)),
            "train_doc_count_used_kept": int(len(kept_rows)),
            "skipped_short_docs": int(skipped_short),
            "outlier_rule": {
                "kind": "median_mad_per_slot",
                "z_threshold": float(z_thr),
                "pct_fallback_if_mad0": float(pct_tol),
            },
            "slot_stats": slot_stats,
            "mean_sizes": mean_sizes,
            "std_sizes": std_sizes,
            "selection_rule": "nearest_contiguous_pair_by_normalized_mean_distance",
        },
    }


def _build_slot_size_models(
    *,
    slot_templates: list[list[TemplateRef]],
    bytes_ref_mode: str,
) -> list[SlotSizeModel]:
    mode = str(bytes_ref_mode).strip().lower()
    if mode not in {"robust", "legacy"}:
        raise ValueError(f"invalid bytes_ref_mode: {bytes_ref_mode!r} (expected: robust|legacy)")

    out: list[SlotSizeModel] = []
    tpl_docs: dict[str, fitz.Document] = {}
    try:
        for slot, grp in enumerate(slot_templates, start=1):
            if not grp:
                continue
            sample_sizes: list[int] = []
            for t in grp:
                pdfp = str(Path(t.pdf_path).resolve())
                if pdfp not in tpl_docs:
                    tpl_docs[pdfp] = fitz.open(pdfp)
                d = tpl_docs[pdfp]
                sample_sizes.append(int(len(_one_page_pdf_bytes(d, page_index=int(t.page_index)))))

            rep = grp[0]
            if mode == "legacy":
                model_size = int(sample_sizes[0]) if sample_sizes else 0
                median_size = float(model_size)
                mad_size = 0.0
            else:
                median_size = _median_float(sample_sizes)
                mad_size = _mad_float(sample_sizes, center=median_size)
                model_size = int(round(median_size)) if sample_sizes else 0

            out.append(
                SlotSizeModel(
                    slot=int(slot),
                    template_page_index=int(rep.page_index),
                    model_size=int(model_size),
                    median_size=float(median_size),
                    mad_size=float(mad_size),
                    sample_count=int(len(sample_sizes)),
                )
            )
        return out
    finally:
        for d in tpl_docs.values():
            try:
                d.close()
            except Exception:
                pass


def _run_doc_extract(
    *,
    root: Path,
    templates: list[TemplateRef],
    label: str,
    method: str,
    specs: list[str],
    save_dir: Path | None,
    top_n: int = 5,
    min_p1: float | None = None,
    min_p2: float | None = None,
    bytes_ref_mode: str = "robust",
    kmeans_reference: dict[str, Any] | None = None,
    mean2_reference: dict[str, Any] | None = None,
    emit_details: bool = True,
    candidate_position: tuple[int, int] | None = None,
    console: Console | None = None,
    log_sink: Callable[[str, str | None], None] | None = None,
) -> list[dict[str, Any]]:
    label = str(label).strip()
    if not label:
        raise ValueError("label is empty")

    tpls = [t for t in templates if t.label == label]
    if not tpls:
        raise ValueError(f"no templates for label: {label}")
    templates_seq = sorted(tpls, key=lambda t: int(t.page_index))
    slot_templates = _group_templates_by_slot(templates_seq)
    templates_seq = _slot_representatives(slot_templates)
    K = int(len(templates_seq))
    if K <= 0:
        raise ValueError(f"no template slots for label: {label}")
    method = str(method).strip().lower()

    method_labels = {
        "bytes": "BYTES (size-only)",
        "tlsh": "TLSH (arquivo/janela)",
        "stream": "STREAM (/Contents)",
        "text": "TEXTO",
        "image": "IMAGEM",
    }
    method_analysis = {
        "bytes": "size-bytes",
        "tlsh": "tlsh-bytes",
        "stream": "stream-contents",
        "text": "texto",
        "image": "imagem-dhash",
    }
    method_colors = {
        "bytes": "#5f6f7f",
        "tlsh": "#6a6f7a",
        "stream": "#5f7a6f",
        "text": "#b87838",
        "image": "#75657a",
    }

    def _emit(text: str = "", style: str | None = None) -> None:
        if not emit_details:
            return
        if log_sink is not None:
            log_sink(text, style)
            return
        if style and console is not None:
            console.print(text, style=style)
            return
        print(text)

    def _p(text: str = "") -> None:
        if not emit_details:
            return
        color = method_colors.get(method, "white")
        _emit(text, color)

    def _pc(text: str, *, bold: bool = False) -> None:
        if not emit_details:
            return
        color = method_colors.get(method, "white")
        style = f"bold {color}" if bold else color
        _emit(text, style)

    def _window_payload(w: BestWindow) -> dict[str, Any]:
        page_indices: list[int] = []
        for r in w.per_page:
            pi = r.get("page_index")
            if isinstance(pi, int):
                page_indices.append(int(pi))
        if not page_indices:
            if int(w.start_page) <= int(w.end_page):
                page_indices = list(range(int(w.start_page), int(w.end_page) + 1))
            else:
                page_indices = [int(w.start_page), int(w.end_page)]
        sidx = int(min(page_indices)) if page_indices else int(w.start_page)
        eidx = int(max(page_indices)) if page_indices else int(w.end_page)
        payload: dict[str, Any] = {
            "start_page_index": sidx,
            "end_page_index": eidx,
            "start_page": sidx + 1,
            "end_page": eidx + 1,
            "page_indices": page_indices,
            "total": float(w.total),
            "per_page": list(w.per_page),
        }
        if isinstance(w.meta, dict) and w.meta:
            payload["meta"] = dict(w.meta)
        return payload

    kmeans_model_ctx: dict[str, Any] | None = None
    kmeans_summary: dict[str, Any] | None = None
    mean2_summary: dict[str, Any] | None = None
    bytes_mode_active = method in {"bytes", "byte", "size", "size_only"}
    if bytes_mode_active and str(bytes_ref_mode).strip().lower() == "kmeans":
        if not isinstance(kmeans_reference, dict):
            raise ValueError("bytes_ref_mode=kmeans requires explicit kmeans_reference (no implicit fallback)")
        slot_size_models = list(kmeans_reference.get("slot_size_models") or [])
        kmeans_model_ctx = kmeans_reference.get("kmeans_model") if isinstance(kmeans_reference.get("kmeans_model"), dict) else None
        kmeans_summary = kmeans_reference.get("summary") if isinstance(kmeans_reference.get("summary"), dict) else None
        if not slot_size_models or kmeans_model_ctx is None:
            raise ValueError("invalid kmeans_reference payload (missing slot_size_models/kmeans_model)")
    elif bytes_mode_active and str(bytes_ref_mode).strip().lower() == "mean2":
        if not isinstance(mean2_reference, dict):
            raise ValueError("bytes_ref_mode=mean2 requires explicit mean2_reference (no implicit fallback)")
        slot_size_models = list(mean2_reference.get("slot_size_models") or [])
        mean2_summary = mean2_reference.get("summary") if isinstance(mean2_reference.get("summary"), dict) else None
        if not slot_size_models:
            raise ValueError("invalid mean2_reference payload (missing slot_size_models)")
    else:
        fallback_mode = str(bytes_ref_mode).strip().lower()
        if fallback_mode not in {"robust", "legacy"}:
            fallback_mode = "robust"
        slot_size_models = _build_slot_size_models(slot_templates=slot_templates, bytes_ref_mode=fallback_mode)
    model_sizes = [int(sm.model_size) for sm in slot_size_models]
    if len(model_sizes) != K:
        raise ValueError(
            f"template slot size mismatch for label={label}: slots={K} size_models={len(model_sizes)}"
        )
    model_pdf_names = [Path(t.pdf_path).name for t in templates_seq]
    model_display = model_pdf_names[0] if model_pdf_names else "-"
    if model_pdf_names:
        uniq_model_names = sorted(set(model_pdf_names))
        if len(uniq_model_names) > 1:
            model_display = f"{uniq_model_names[0]} (+{len(uniq_model_names)-1} arquivo(s))"
    slot_reference: list[dict[str, Any]] = []
    for pos, sm in enumerate(slot_size_models):
        grp = slot_templates[pos] if pos < len(slot_templates) else []
        slot_reference.append(
            {
                "slot": int(sm.slot),
                "template_page_index": int(sm.template_page_index),
                "model_size": int(sm.model_size),
                "median_size": float(sm.median_size),
                "mad_size": float(sm.mad_size),
                "sample_count": int(sm.sample_count),
                "templates": [
                    {
                        "id": str(t.id),
                        "pdf": str(Path(t.pdf_path)),
                        "page_index": int(t.page_index),
                    }
                    for t in grp
                ],
            }
        )

    # Template features (method-specific).
    tpl_texts: list[str] | None = None
    tpl_stream_fps: list[object | None] | None = None
    tpl_dhashes: list[object | None] | None = None

    image_dpi = int(os.getenv("PDFCATCH_IMAGE_DPI", "72") or "72")
    image_hash_size = int(os.getenv("PDFCATCH_IMAGE_HASH_SIZE", "8") or "8")
    max_dist = int(os.getenv("PDFCATCH_PAGE_BYTES_PREFILTER_MAXDIST", "200") or "200")
    top_n = max(1, int(top_n))
    bytes_min_any = float(_env_float("PDFCATCH_BYTES_MIN_ANY", 0.40))
    bytes_min_p1 = float(min_p1) if min_p1 is not None else float(_env_float("PDFCATCH_BYTES_MIN_P1", 0.60))
    bytes_min_p2 = float(min_p2) if min_p2 is not None else float(_env_float("PDFCATCH_BYTES_MIN_P2", 0.40))

    # TLSH model window (method-specific), computed once per label+run.
    model_win_tlsh: str | None = None
    model_win_size: int | None = None
    model_win_sha256: str | None = None
    if method == "tlsh" and _tlsh is not None:
        model_win_doc = fitz.open()
        tpl_docs: dict[str, fitz.Document] = {}
        try:
            for t in templates_seq:
                pdfp = str(Path(t.pdf_path).resolve())
                if pdfp not in tpl_docs:
                    tpl_docs[pdfp] = fitz.open(pdfp)
                model_win_doc.insert_pdf(tpl_docs[pdfp], from_page=int(t.page_index), to_page=int(t.page_index))
            model_win_bytes = model_win_doc.tobytes(**_SAVE_KW)
            model_win_size = int(len(model_win_bytes))
            model_win_sha256 = _sha256_bytes(model_win_bytes)
            h = _tlsh.hash(model_win_bytes)
            model_win_tlsh = None if (not h or h == "TNULL") else str(h)
        finally:
            try:
                model_win_doc.close()
            except Exception:
                pass
            for d in tpl_docs.values():
                try:
                    d.close()
                except Exception:
                    pass

    # Expand specs -> PDFs (ordered, with duplicates removed by absolute path).
    pdfs: list[Path] = []
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        expanded, meta = _expand_spec_to_pdfs(root, spec)
        sources.append(meta)
        for p in expanded:
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            pdfs.append(p)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_label_dir: Path | None = None
    if save_dir is not None:
        out_label_dir = (save_dir / label).resolve()
        out_label_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    total_pdfs = len(pdfs)
    for local_idx, cand_pdf in enumerate(pdfs, start=1):
        file_ord, file_total = (local_idx, total_pdfs)
        if candidate_position is not None:
            file_ord, file_total = int(candidate_position[0]), int(candidate_position[1])

        _pc(f"Avaliando similaridade por {method_analysis.get(method, method)}...", bold=True)
        _p("")
        _pc(f"ARQUIVO = {cand_pdf.name} ({file_ord}/{file_total})", bold=True)
        _pc(f"LABEL   = {label}", bold=True)
        _pc(f"METODO  = {method_labels.get(method, method.upper())}", bold=True)
        _p("")
        _pc(f"MODELO  = {model_display}", bold=True)
        _p("")
        for pos, t in enumerate(templates_seq):
            if method in {"bytes", "byte", "size", "size_only"}:
                sm = slot_size_models[pos]
                _p(
                    f"page_idx: {int(t.page_index):>4} | size_model={int(model_sizes[pos]):>8}"
                    f" | median={float(sm.median_size):.1f} | mad={float(sm.mad_size):.1f}"
                    f" | n={int(sm.sample_count):>2} | score=1.0000"
                )
            else:
                _p(f"page_idx: {int(t.page_index):>4} | score=1.0000")
        if method in {"bytes", "byte", "size", "size_only"}:
            slot_rules: list[str] = []
            for pos in range(K):
                if pos == 0:
                    slot_rules.append(f"p1>{bytes_min_p1:.2f}")
                elif pos == 1:
                    slot_rules.append(f"p2>={bytes_min_p2:.2f}")
                else:
                    slot_rules.append(f"p{pos+1}>={bytes_min_any:.2f}")
            _p(f"bytes_ref_mode: {bytes_ref_mode}")
            if str(bytes_ref_mode).strip().lower() == "kmeans" and isinstance(kmeans_summary, dict):
                _p(
                    "kmeans_target: "
                    f"cluster={kmeans_summary.get('target_cluster')} "
                    f"| windows={kmeans_summary.get('train_window_count')} "
                    f"| tips={kmeans_summary.get('tip_term_count')}"
                )
                _p(f"kmeans_rule: {kmeans_summary.get('target_cluster_rule')}")
                if kmeans_summary.get("anchor_source"):
                    _p(f"kmeans_anchor_source: {kmeans_summary.get('anchor_source')}")
            if str(bytes_ref_mode).strip().lower() == "mean2" and isinstance(mean2_summary, dict):
                _p(
                    "mean2_ref: "
                    f"train_raw={mean2_summary.get('train_doc_count_used_raw')} "
                    f"| kept={mean2_summary.get('train_doc_count_used_kept')}"
                )
                _p(f"mean2_rule: {mean2_summary.get('selection_rule')}")
            if str(bytes_ref_mode).strip().lower() == "mean2":
                _p("min_scores: (off in mean2 mode; nearest distance only)")
            else:
                _p(f"min_scores: any>={bytes_min_any:.2f} | " + " | ".join(slot_rules))
        if method == "tlsh" and model_win_tlsh:
            _p(
                f"janela_modelo: size_bytes={int(model_win_size or 0)} "
                f"| sha256={model_win_sha256 or '-'} | tlsh={model_win_tlsh}"
            )
            _p(f"normalizacao_tlsh: escala={int(max_dist)} | formula=1/(1 + dist/escala)")
        _p("")

        sha = _sha256_file(cand_pdf)
        ranked_windows: list[BestWindow] = []
        ranked_extras: list[dict[str, Any]] = []
        extra_best: dict[str, Any] = {}
        candidate_page_count: int | None = None
        window_count: int | None = None

        if method in {"bytes", "byte", "size", "size_only"}:
            if str(bytes_ref_mode).strip().lower() == "mean2":
                ranked_windows, candidate_page_count, window_count = _rank_windows_by_mean2(
                    candidate_pdf=cand_pdf,
                    mean_page_sizes=[float(v) for v in model_sizes],
                )
            else:
                ranked_windows, candidate_page_count, window_count = _rank_windows_by_size(
                    candidate_pdf=cand_pdf,
                    model_page_sizes=model_sizes,
                    min_p1_override=bytes_min_p1,
                    min_p2_override=bytes_min_p2,
                    kmeans_ctx=kmeans_model_ctx,
                )
        elif method == "tlsh":
            if _tlsh is None:
                ranked_windows = []
                extra_best = {"status": "skipped", "reason": "tlsh_unavailable (install py-tlsh)"}
            elif not model_win_tlsh:
                ranked_windows = []
                extra_best = {"status": "skipped", "reason": "model_tlsh_null"}
            else:
                model_tlsh = str(model_win_tlsh)
                model_size = int(model_win_size or 0)
                cand_doc = fitz.open(str(cand_pdf))
                ranked_pairs: list[tuple[BestWindow, dict[str, Any]]] = []
                try:
                    N = int(cand_doc.page_count)
                    candidate_page_count = N
                    window_count = max(0, N - K + 1)
                    if N >= K:
                        for start in range(0, N - K + 1):
                            end = int(start + K - 1)
                            out = fitz.open()
                            try:
                                out.insert_pdf(cand_doc, from_page=int(start), to_page=int(end))
                                win_bytes = out.tobytes(**_SAVE_KW)
                            finally:
                                try:
                                    out.close()
                                except Exception:
                                    pass
                            win_tlsh = _tlsh.hash(win_bytes)
                            if not win_tlsh or win_tlsh == "TNULL":
                                continue
                            dist = int(_tlsh.diffxlen(str(win_tlsh), str(model_tlsh)))
                            scale = float(max(1, int(max_dist)))
                            sim = float(1.0 / (1.0 + (float(dist) / scale)))
                            bw = BestWindow(start_page=int(start), end_page=int(end), total=float(sim), per_page=[])
                            extra = {
                                "tlsh_similarity": float(sim),
                                "tlsh_distance": int(dist),
                                "window_size_bytes": int(len(win_bytes)),
                                "model_size_bytes": int(model_size),
                                "window_size_similarity": float(_size_similarity(int(model_size), int(len(win_bytes))))
                                if model_size > 0
                                else None,
                            }
                            ranked_pairs.append((bw, extra))
                finally:
                    try:
                        cand_doc.close()
                    except Exception:
                        pass
                ranked_pairs.sort(
                    key=lambda it: (
                        -float(it[0].total),
                        int(it[1].get("tlsh_distance") if it[1].get("tlsh_distance") is not None else 10**9),
                    )
                )
                ranked_windows = [it[0] for it in ranked_pairs]
                ranked_extras = [it[1] for it in ranked_pairs]
                if ranked_extras:
                    extra_best = ranked_extras[0]
        elif method == "stream":
            if tpl_stream_fps is None:
                tpl_stream_fps = []
                tpl_pike_docs: dict[str, object] = {}
                try:
                    for t in templates_seq:
                        pdfp = str(Path(t.pdf_path).resolve())
                        if pdfp not in tpl_pike_docs:
                            tpl_pike_docs[pdfp] = open_pikepdf(pdfp)
                        pdoc = tpl_pike_docs[pdfp]
                        try:
                            tpl_stream_fps.append(fingerprint_page_contents(pdoc, page_index=int(t.page_index)))
                        except Exception:
                            tpl_stream_fps.append(None)
                finally:
                    for d in tpl_pike_docs.values():
                        try:
                            d.close()
                        except Exception:
                            pass

            cand_pike = None
            try:
                cand_pike = open_pikepdf(cand_pdf)
                cand_fps: dict[int, object | None] = {}
                cand_doc = fitz.open(str(cand_pdf))
                try:
                    N = int(cand_doc.page_count)
                    candidate_page_count = N
                    window_count = max(0, N - K + 1)
                finally:
                    try:
                        cand_doc.close()
                    except Exception:
                        pass

                if N >= K and tpl_stream_fps and all(fp is not None for fp in tpl_stream_fps):
                    for start in range(0, N - K + 1):
                        per: list[dict[str, Any]] = []
                        scores: list[float] = []
                        ok = True
                        for pos in range(K):
                            pi = int(start + pos)
                            if pi not in cand_fps:
                                try:
                                    cand_fps[pi] = fingerprint_page_contents(cand_pike, page_index=pi)
                                except Exception:
                                    cand_fps[pi] = None
                            a = tpl_stream_fps[pos]
                            b = cand_fps[pi]
                            if a is None or b is None:
                                ok = False
                                break
                            sc = float(stream_similarity(b, a))
                            per.append({"page_index": pi, "score": sc})
                            scores.append(sc)
                        if ok and scores:
                            total = float(sum(scores) / float(len(scores)))
                            ranked_windows.append(
                                BestWindow(start_page=int(start), end_page=int(start + K - 1), total=total, per_page=per)
                            )
            finally:
                if cand_pike is not None:
                    try:
                        cand_pike.close()
                    except Exception:
                        pass
            ranked_windows.sort(key=lambda w: w.total, reverse=True)
        elif method == "text":
            if tpl_texts is None:
                tpl_texts = []
                tpl_docs: dict[str, fitz.Document] = {}
                try:
                    for t in templates_seq:
                        pdfp = str(Path(t.pdf_path).resolve())
                        if pdfp not in tpl_docs:
                            tpl_docs[pdfp] = fitz.open(pdfp)
                        fdoc = tpl_docs[pdfp]
                        tpl_texts.append(fdoc.load_page(int(t.page_index)).get_text("text") or "")
                finally:
                    for d in tpl_docs.values():
                        try:
                            d.close()
                        except Exception:
                            pass

            cand_doc = fitz.open(str(cand_pdf))
            try:
                N = int(cand_doc.page_count)
                candidate_page_count = N
                window_count = max(0, N - K + 1)
                if N >= K and tpl_texts:
                    cand_texts: dict[int, str] = {}
                    for start in range(0, N - K + 1):
                        per: list[dict[str, Any]] = []
                        scores: list[float] = []
                        for pos in range(K):
                            pi = int(start + pos)
                            if pi not in cand_texts:
                                cand_texts[pi] = cand_doc.load_page(pi).get_text("text") or ""
                            s = score_text(tpl_texts[pos], cand_texts[pi])
                            per.append(
                                {
                                    "page_index": pi,
                                    "hybrid": float(s.hybrid),
                                    "order": float(s.order),
                                    "cosine": float(s.cosine),
                                    "jaccard": float(s.jaccard),
                                }
                            )
                            scores.append(float(s.hybrid))
                        total = float(sum(scores) / float(len(scores) or 1))
                        ranked_windows.append(
                            BestWindow(start_page=int(start), end_page=int(start + K - 1), total=total, per_page=per)
                        )
            finally:
                try:
                    cand_doc.close()
                except Exception:
                    pass
            ranked_windows.sort(key=lambda w: w.total, reverse=True)
        elif method == "image":
            if tpl_dhashes is None:
                tpl_dhashes = []
                tpl_docs: dict[str, fitz.Document] = {}
                try:
                    for t in templates_seq:
                        pdfp = str(Path(t.pdf_path).resolve())
                        if pdfp not in tpl_docs:
                            tpl_docs[pdfp] = fitz.open(pdfp)
                        fdoc = tpl_docs[pdfp]
                        try:
                            tpl_dhashes.append(
                                page_dhash(
                                    fdoc,
                                    page_index=int(t.page_index),
                                    hash_size=int(image_hash_size),
                                    dpi=int(image_dpi),
                                )
                            )
                        except Exception:
                            tpl_dhashes.append(None)
                finally:
                    for d in tpl_docs.values():
                        try:
                            d.close()
                        except Exception:
                            pass

            cand_doc = fitz.open(str(cand_pdf))
            try:
                N = int(cand_doc.page_count)
                candidate_page_count = N
                window_count = max(0, N - K + 1)
                if N >= K and tpl_dhashes and all(d is not None for d in tpl_dhashes):
                    cand_hashes: dict[int, object | None] = {}
                    for start in range(0, N - K + 1):
                        per: list[dict[str, Any]] = []
                        scores: list[float] = []
                        ok = True
                        for pos in range(K):
                            pi = int(start + pos)
                            if pi not in cand_hashes:
                                try:
                                    cand_hashes[pi] = page_dhash(
                                        cand_doc,
                                        page_index=pi,
                                        hash_size=int(image_hash_size),
                                        dpi=int(image_dpi),
                                    )
                                except Exception:
                                    cand_hashes[pi] = None
                            a = tpl_dhashes[pos]
                            b = cand_hashes[pi]
                            if a is None or b is None:
                                ok = False
                                break
                            sc = float(dhash_score(a, b))
                            per.append({"page_index": pi, "score": sc})
                            scores.append(sc)
                        if ok and scores:
                            total = float(sum(scores) / float(len(scores)))
                            ranked_windows.append(
                                BestWindow(start_page=int(start), end_page=int(start + K - 1), total=total, per_page=per)
                            )
            finally:
                try:
                    cand_doc.close()
                except Exception:
                    pass
            ranked_windows.sort(key=lambda w: w.total, reverse=True)
        else:
            raise ValueError(f"unknown method: {method}")

        best = ranked_windows[0] if ranked_windows else None
        top_windows = ranked_windows[:top_n]
        if ranked_extras and best is not None:
            extra_best = ranked_extras[0]

        analyzed = int(window_count or 0)
        shown_top = int(min(max(1, int(top_n)), len(top_windows) if ranked_windows else max(1, int(top_n))))
        _pc(f"PAGES   = {analyzed}", bold=True)
        _p(f"analise: janelas avaliadas={analyzed} | exibindo_top={shown_top}")
        _pc(f"RESULTADO top={top_n}:", bold=True)

        if best is None:
            _p("(sem resultado)")
            _p("")
            reason = f"no_result (pages < K ({K}) or missing features)"
            if (
                method in {"bytes", "byte", "size", "size_only"}
                and (int(candidate_page_count or 0) >= int(K))
                and str(bytes_ref_mode).strip().lower() != "mean2"
            ):
                reason = (
                    "no_window_passed_min_scores "
                    f"(any>={bytes_min_any:.2f}, p1>{bytes_min_p1:.2f}, p2>={bytes_min_p2:.2f})"
                )
            row = {
                "label": label,
                "method": method,
                "candidate_pdf": str(cand_pdf),
                "sha256": sha,
                "status": "skipped",
                "reason": extra_best.get("reason") or reason,
            }
            results.append(row)
            continue

        for pos, tw in enumerate(top_windows, start=1):
            _pc(f"TOP {pos}", bold=True)
            if method == "tlsh":
                ex = ranked_extras[pos - 1] if pos - 1 < len(ranked_extras) else {}
                _p(
                    f"janela_idx0: {int(tw.start_page):04d}-{int(tw.end_page):04d} "
                    f"| tlsh_dist={ex.get('tlsh_distance', '-')}"
                )
            for slot_idx, r in enumerate(tw.per_page, start=1):
                pi = int(r.get("page_index", -1))
                if "size_bytes" in r:
                    if str(bytes_ref_mode).strip().lower() == "mean2":
                        dnorm = r.get("distance_norm")
                        if isinstance(dnorm, (int, float)):
                            _p(
                                f"page_idx: {pi:>4} | size_bytes={int(r['size_bytes']):>8} "
                                f"| mean={float(r.get('mean_size') or 0):.1f} "
                                f"| dist_norm={float(dnorm):.6f} | score={float(r['score']):.4f}"
                            )
                        else:
                            _p(f"page_idx: {pi:>4} | size_bytes={int(r['size_bytes']):>8} | score={float(r['score']):.4f}")
                    else:
                        if slot_idx == 1:
                            slot_cmp = ">"
                            slot_min = float(bytes_min_p1)
                        elif slot_idx == 2:
                            slot_cmp = ">="
                            slot_min = float(bytes_min_p2)
                        else:
                            slot_cmp = ">="
                            slot_min = float(bytes_min_any)
                        _p(
                            f"page_idx: {pi:>4} | size_bytes={int(r['size_bytes']):>8} | score={float(r['score']):.4f}"
                            f" | min(slot{slot_idx}){slot_cmp}{slot_min:.2f}"
                        )
                elif "hybrid" in r:
                    _p(
                        f"page_idx: {pi:>4} | score={float(r['hybrid']):.4f} "
                        f"| order={float(r['order']):.4f} cosine={float(r['cosine']):.4f} jaccard={float(r['jaccard']):.4f}"
                    )
                else:
                    _p(f"page_idx: {pi:>4} | score={float(r.get('score', 0.0)):.4f}")
            if method in {"bytes", "byte", "size", "size_only"} and isinstance(tw.meta, dict):
                size_total = tw.meta.get("size_total")
                km_sim = tw.meta.get("kmeans_similarity")
                km_dist = tw.meta.get("kmeans_distance")
                near_c = tw.meta.get("nearest_cluster")
                tgt_c = tw.meta.get("target_cluster")
                if isinstance(km_sim, (int, float)) and isinstance(km_dist, (int, float)):
                    _p(
                        "kmeans: "
                        f"size_total={float(size_total or 0.0):.4f} "
                        f"| sim={float(km_sim):.4f} | dist={float(km_dist):.4f} "
                        f"| cluster={near_c} target={tgt_c}"
                    )
                mean_dist = tw.meta.get("mean_distance")
                if isinstance(mean_dist, (int, float)) and str(bytes_ref_mode).strip().lower() == "mean2":
                    _p(f"mean2: mean_distance={float(mean_dist):.6f} | nearest_pair_distance")
            _pc(f"final_sc: {float(tw.total):.4f}", bold=(pos == 1))
            _p("")

        best_page_indices: list[int] = []
        for r in best.per_page:
            pi = r.get("page_index")
            if isinstance(pi, int):
                best_page_indices.append(int(pi))
        if not best_page_indices:
            if int(best.start_page) <= int(best.end_page):
                best_page_indices = list(range(int(best.start_page), int(best.end_page) + 1))
            else:
                best_page_indices = [int(best.start_page), int(best.end_page)]
        start = int(min(best_page_indices)) if best_page_indices else int(best.start_page)
        end = int(max(best_page_indices)) if best_page_indices else int(best.end_page)
        row: dict[str, Any] = {
            "label": label,
            "method": method,
            "method_score": float(best.total),
            "K": K,
            "bytes_ref_mode": bytes_ref_mode,
            "candidate_pdf": str(cand_pdf),
            "sha256": sha,
            "candidate_page_count": candidate_page_count,
            "window_count": window_count,
            "model_reference": {
                "label": label,
                "bytes_ref_mode": bytes_ref_mode,
                "slots": slot_reference,
            },
            "best_window": {
                "start_page_index": start,
                "end_page_index": end,
                "start_page": start + 1,
                "end_page": end + 1,
                "page_indices": list(best_page_indices),
                "per_page": best.per_page,
            },
            "top_windows": [_window_payload(w) for w in top_windows],
            "all_windows": [_window_payload(w) for w in ranked_windows],
            "status": "ok",
        }
        if method in {"bytes", "byte", "size", "size_only"} and isinstance(kmeans_summary, dict):
            row["model_reference"]["kmeans"] = kmeans_summary
        if method in {"bytes", "byte", "size", "size_only"} and isinstance(mean2_summary, dict):
            row["model_reference"]["mean2"] = mean2_summary

        # Add opaque file identity for best window (no content parsing).
        cand_doc = fitz.open(str(cand_pdf))
        try:
            win_data = _pages_pdf_bytes(cand_doc, page_indices=list(best_page_indices))
            row["best_window"]["window_size_bytes"] = int(len(win_data))
            row["best_window"]["window_sha256"] = _sha256_bytes(win_data)
        finally:
            try:
                cand_doc.close()
            except Exception:
                pass

        if method in {"bytes", "byte", "size", "size_only"}:
            bw_meta = best.meta if isinstance(best.meta, dict) else {}
            if isinstance(bw_meta.get("size_total"), (int, float)):
                row["best_window"]["size_total"] = float(bw_meta.get("size_total"))
            else:
                row["best_window"]["size_total"] = float(best.total)
            if str(bytes_ref_mode).strip().lower() == "kmeans":
                row["best_window"]["kmeans_total"] = float(best.total)
                row["best_window"]["kmeans_similarity"] = (
                    float(bw_meta.get("kmeans_similarity")) if isinstance(bw_meta.get("kmeans_similarity"), (int, float)) else None
                )
                row["best_window"]["kmeans_distance"] = (
                    float(bw_meta.get("kmeans_distance")) if isinstance(bw_meta.get("kmeans_distance"), (int, float)) else None
                )
                row["best_window"]["kmeans_nearest_cluster"] = bw_meta.get("nearest_cluster")
                row["best_window"]["kmeans_target_cluster"] = bw_meta.get("target_cluster")
                row["best_window"]["kmeans_is_target_cluster"] = bool(bw_meta.get("is_target_cluster"))
            if str(bytes_ref_mode).strip().lower() == "mean2":
                row["best_window"]["mean2_total"] = float(best.total)
                row["best_window"]["mean2_distance"] = (
                    float(bw_meta.get("mean_distance")) if isinstance(bw_meta.get("mean_distance"), (int, float)) else None
                )
                row["best_window"]["mean2_selection_rule"] = "nearest_contiguous_pair_by_normalized_mean_distance"
            row["best_window"]["size_rules"] = {
                "min_any": float(bytes_min_any),
                "min_p1_strict_gt": float(bytes_min_p1),
                "min_p2_ge": float(bytes_min_p2),
            }
        elif method == "tlsh":
            row["best_window"]["tlsh_total"] = float(best.total)
            row["best_window"]["details"] = extra_best
        elif method == "stream":
            row["best_window"]["stream_total"] = float(best.total)
        elif method == "text":
            row["best_window"]["text_total"] = float(best.total)
        elif method == "image":
            row["best_window"]["image_total"] = float(best.total)

        if out_label_dir is not None:
            idx_tok = _idx0_compact(best_page_indices)
            pages_tok = _pages_compact(best_page_indices)
            out_name = (
                f"{_safe_stem(cand_pdf.stem)}__{sha[:12]}__{label}__m{_safe_stem(method)}__idx{idx_tok}"
                f"__pag{pages_tok}__{ts}.pdf"
            )
            out_pdf = (out_label_dir / out_name).resolve()
            cand_doc = fitz.open(str(cand_pdf))
            try:
                _export_pages_pdf(cand_doc, page_indices=list(best_page_indices), out_path=out_pdf)
            finally:
                try:
                    cand_doc.close()
                except Exception:
                    pass
            row["exported_pdf"] = str(out_pdf)
            _p(f"exported: {out_pdf}")
            _p("")

        results.append(row)

    # Write a per-label manifest.
    if out_label_dir is not None:
        manifest = {
            "label": label,
            "method": method,
            "K": K,
            "bytes_ref_mode": bytes_ref_mode,
            "sources": sources,
            "model_page_sizes": model_sizes,
            "model_slots": slot_reference,
            "kmeans_reference": kmeans_summary if isinstance(kmeans_summary, dict) else None,
            "mean2_reference": mean2_summary if isinstance(mean2_summary, dict) else None,
            "outputs": results,
        }
        (out_label_dir / f"manifest__{label}__m{_safe_stem(method)}__{ts}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli",
        description="pdfcatch CLI (local/offline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_app_version()}",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    doc = sub.add_parser("doc", aliases=["document"], help="Extract documents by label (size-only window selection).")
    doc.add_argument("-d", "--despacho", action="append", default=[], help="Spec for DESPACHO PDFs (ex.: :Q4-10)")
    doc.add_argument("-c", "--certidao", action="append", default=[], help="Spec for CERTIDAO_CM PDFs (ex.: :Q3-8)")
    doc.add_argument(
        "-m",
        "--method",
        action="append",
        nargs="+",
        default=[],
        help="Metodo: bytes|tlsh|stream|text|image|all (default: bytes). "
        "Aceita multiplos: --method bytes stream ou repetido: -m bytes -m stream.",
    )
    doc.add_argument("--all", action="store_true", help="Roda todos os metodos em sequencia (bytes, tlsh, stream, text, image).")
    doc.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Quantidade de candidatos no ranking por metodo (default: PDFCATCH_TOP_N ou 5).",
    )
    doc.add_argument(
        "--min-p1",
        type=float,
        default=None,
        help="Override do minimo de score para a pagina 1 no metodo bytes (0..1).",
    )
    doc.add_argument(
        "--min-p2",
        type=float,
        default=None,
        help="Override do minimo de score para a pagina 2 no metodo bytes (0..1).",
    )
    doc.add_argument(
        "--bytes-ref-mode",
        default=None,
        choices=["robust", "legacy", "kmeans", "mean2"],
        help=(
            "Referencia do metodo bytes: robust=mediana/MAD por slot; "
            "legacy=template unico por slot; "
            "kmeans=clusterizacao de janelas por bytes com cluster-alvo textual explicito; "
            "mean2=media de bytes por slot (p1,p2) com exclusao de outliers."
        ),
    )
    doc.add_argument(
        "--bytes-train-spec",
        action="append",
        default=[],
        help=(
            "Spec(s) de treino para bytes_ref_mode=mean2 (ex.: :D, /dir/despachos). "
            "Se vazio, usa PDFCATCH_BYTES_TRAIN_SPEC; se vazio, usa as specs do proprio lote."
        ),
    )
    doc.add_argument(
        "--save-files",
        default=None,
        help="Directory to write extracted PDFs (optional; ex.: ./docs_extracted)",
    )
    doc.add_argument(
        "--templates",
        default=None,
        help="Templates JSON (default: PDFCATCH_TEMPLATES_JSON or configs/templates.json)",
    )
    doc.add_argument(
        "--return",
        dest="return_mode",
        action="store_true",
        help="Salva retorno JSON de runtime (timestamp) em io/out para encadear processamento.",
    )
    doc.add_argument(
        "--io-dir",
        default=None,
        help="Diretorio para retorno JSON do --return (default: io/out).",
    )
    doc.add_argument(
        "--web",
        action="store_true",
        help="After detection, open Web UI with selected input/output dirs and thumbnails enabled.",
    )
    doc.add_argument("--dry-run", action="store_true", help="Only show which PDFs would be processed.")

    xtext = sub.add_parser(
        "extract-text",
        aliases=["alt-extract"],
        help="Text extraction by before/after anchors only.",
    )
    xtext.add_argument(
        "--pdf",
        required=False,
        help="PDF source spec (alias/path/file), e.g.: :Q8-8, :Q1-20, /abs/file.pdf, /abs/dir",
    )
    xtext.add_argument(
        "--from-return",
        default=None,
        help="JSON gerado pelo detector (--return). Use 'latest' para o mais recente em io/out.",
    )
    xtext.add_argument(
        "--return",
        dest="return_mode",
        action="store_true",
        help="Salva retorno JSON de runtime (timestamp) em io/out para encadear processamento.",
    )
    xtext.add_argument(
        "--io-dir",
        default=None,
        help="Diretorio para retorno JSON do --return (default: io/out).",
    )
    xtext.add_argument(
        "--anchors",
        default=None,
        help="Anchor rules JSON (default: PDFCATCH_TEXT_ANCHORS_JSON or configs/anchor_text_fields.example.json)",
    )
    xtext.add_argument("--out", default=None, help="Write JSON output to file.")
    xtext.add_argument("--max-chars", type=int, default=0, help="Trim each extracted value to N chars (0=off).")

    def _add_web_open_args(sp: argparse.ArgumentParser, *, section_default: str, include_section: bool = True) -> None:
        sp.add_argument(
            "--pdfs",
            default=None,
            help="PDF source spec (alias/path/file), e.g.: :Q, :Q4-10, ./PDFs, /abs/dir, /abs/file.pdf",
        )
        sp.add_argument(
            "--dir",
            default=None,
            help="Explicit PDF directory (alternative to --pdfs).",
        )
        sp.add_argument(
            "--save-files",
            default=None,
            help="Extracted docs dir for gallery (optional; ex.: ./docs_extracted).",
        )
        sp.add_argument(
            "--limit",
            type=int,
            default=0,
            help="Initial limit for Deteccao in Web UI (0 = sem limite).",
        )
        if include_section:
            sp.add_argument(
                "--section",
                default=section_default,
                choices=[
                    "Deteccao",
                    "Confirmacao",
                    "Studio",
                    "Estatisticas",
                    "Templates",
                    "Extracao",
                    "Classificacao",
                    "Indexacao",
                    "Revisao",
                ],
                help=f"Initial Web UI section (default: {section_default}).",
            )

    web = sub.add_parser("web", aliases=["ui"], help="Open Web UI focused on a PDF directory/workflow.")
    _add_web_open_args(web, section_default="Studio")

    studio = sub.add_parser(
        "studio",
        help="Open the integrated Studio UI (deteccao + studio + estatisticas), including global vector optimization for anchor matching.",
    )
    _add_web_open_args(studio, section_default="Studio", include_section=False)

    tui = sub.add_parser("tui", help="Open terminal TUI for page-by-page monitoring.")
    tui.add_argument("candidate_pdf", nargs="?", help="Candidate PDF path (optional if set in configs/run.env).")
    tui.add_argument(
        "--templates",
        default=None,
        help="Templates JSON (default: PDFCATCH_TEMPLATES_JSON from configs/run.env).",
    )
    tui.add_argument("--top-k", type=int, default=None)
    tui.add_argument("--no-stream", action="store_true")
    tui.add_argument("--no-text", action="store_true")
    tui.add_argument("--no-image", action="store_true")
    tui.add_argument("--image-dpi", type=int, default=None)
    tui.add_argument("--image-hash-size", type=int, default=None)
    tui.add_argument("--stream-prefilter-min", type=float, default=None)
    tui.add_argument("--out", default=None, help="Write final JSON payload.")
    return p


def main(argv: list[str] | None = None) -> int:
    root = _find_repo_root()
    env_path = root / "configs" / "run.env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

    console = Console()
    argv_list = list(argv if argv is not None else sys.argv[1:])
    if not argv_list:
        _print_cli_help(console)
        return 0
    if argv_list[0] in {"-h", "--help"}:
        _print_cli_help(console)
        return 0
    if argv_list[0].lower() in {"help", "ajuda"}:
        topic = argv_list[1] if len(argv_list) > 1 else None
        _print_cli_help(console, topic=topic)
        return 0

    args = _build_parser().parse_args(argv_list)

    if args.cmd in {"extract-text", "alt-extract"}:
        anchors_path = str(
            args.anchors
            or os.getenv("PDFCATCH_TEXT_ANCHORS_JSON")
            or "configs/anchor_text_fields.example.json"
        )
        anchors_json = _resolve(root, anchors_path)
        if not anchors_json.exists():
            print(f"error: anchors file not found: {anchors_json}", file=sys.stderr)
            return 2
        try:
            rules = load_anchor_rules(anchors_json)
        except Exception as e:
            print(f"error: failed to load anchor rules: {e}", file=sys.stderr)
            return 2

        pdfs: list[Path] = []
        meta: dict[str, Any] = {}
        from_return = str(getattr(args, "from_return", None) or "").strip()
        if not from_return and not args.pdf:
            from_return = "latest"
        if from_return:
            try:
                if from_return.lower() == "latest":
                    io_dir = _resolve(root, "io/out")
                    cands = sorted(io_dir.glob("detector_return__*.json"))
                    if not cands:
                        legacy_dir = _resolve(root, "io")
                        cands = sorted(legacy_dir.glob("detector_return__*.json"))
                    if not cands:
                        raise FileNotFoundError(f"no detector_return json in {io_dir}")
                    ret_path = cands[-1]
                else:
                    ret_path = _resolve(root, from_return)
                obj = json.loads(Path(ret_path).read_text(encoding="utf-8"))
                detected = obj.get("detected") if isinstance(obj, dict) else None
                if isinstance(detected, list) and detected:
                    ps = detected
                else:
                    ps = obj.get("process_summaries") if isinstance(obj, dict) else None
                if not isinstance(ps, list):
                    raise ValueError("process_summaries missing in return JSON")
                seen: set[str] = set()
                for row in ps:
                    if not isinstance(row, dict):
                        continue
                    p = (
                        str(row.get("exported_pdf_final") or "").strip()
                        or str(row.get("candidate_pdf") or "").strip()
                    )
                    if not p:
                        continue
                    rp = Path(p).resolve()
                    if not rp.exists():
                        continue
                    k = str(rp)
                    if k in seen:
                        continue
                    seen.add(k)
                    pdfs.append(rp)
                meta = {"mode": "from_return", "return_json": str(Path(ret_path).resolve()), "selected": len(pdfs)}
            except Exception as e:
                print(f"error: invalid --from-return: {e}", file=sys.stderr)
                return 2
        else:
            if not args.pdf:
                print("error: provide --pdf or --from-return", file=sys.stderr)
                return 2
            try:
                pdfs, meta = _expand_spec_to_pdfs(root, str(args.pdf))
            except Exception as e:
                print(f"error: invalid --pdf spec: {e}", file=sys.stderr)
                return 2
        if not pdfs:
            print("error: no PDFs selected for extraction", file=sys.stderr)
            return 2

        out_rows: list[dict[str, Any]] = []
        for i, p in enumerate(pdfs, start=1):
            r = extract_pdf_by_anchors(p, rules=rules, max_chars=int(args.max_chars or 0))
            out_rows.append(r)
            cov = float(r.get("coverage") or 0.0) * 100.0
            print(f"ARQUIVO {i}/{len(pdfs)}: {Path(str(r.get('pdf') or p)).name} | coverage={cov:0.2f}%")
            for f in (r.get("fields") or []):
                fid = str(f.get("id") or "-")
                ok = bool(f.get("ok"))
                val = str(f.get("value") or "<nao_encontrado>")
                why = str(f.get("reason") or "-")
                print(f"  {fid}: {val} | status={'OK' if ok else 'MISS'} | reason={why}")
            print("")

        payload = {
            "mode": "extract-text-before-after",
            "pdf_spec": (str(args.pdf) if args.pdf else None),
            "spec_meta": meta,
            "anchors_file": str(anchors_json),
            "count": len(out_rows),
            "results": out_rows,
        }
        if args.out:
            outp = _resolve(root, str(args.out))
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"wrote: {outp}")
        if bool(getattr(args, "return_mode", False)):
            ts_ret = datetime.now().strftime("%Y%m%d_%H%M%S")
            io_dir = _resolve(root, str(getattr(args, "io_dir", None) or "io/out"))
            io_dir.mkdir(parents=True, exist_ok=True)
            ret_payload = {
                "schema": "pdfcatch.extrator.return.v1",
                "run_ts": ts_ret,
                "command": "extract-text",
                "cwd": str(root),
                "anchors_file": str(anchors_json),
                "source": {
                    "pdf_spec": (str(args.pdf) if args.pdf else None),
                    "from_return": (from_return or None),
                    "spec_meta": meta,
                },
                "result": payload,
            }
            ret_path = (io_dir / f"extractor_return__{ts_ret}.json").resolve()
            ret_path.write_text(json.dumps(ret_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"return_json: {ret_path}")
        return 0

    if args.cmd in {"web", "ui", "studio"}:
        if args.dir and args.pdfs:
            print("error: use only one of --dir or --pdfs", file=sys.stderr)
            return 2

        active_pdf_dir: Path | None = None
        if args.dir:
            p = _resolve(root, str(args.dir))
            if not p.exists() or not p.is_dir():
                print(f"error: invalid --dir: {p}", file=sys.stderr)
                return 2
            active_pdf_dir = p.resolve()
        elif args.pdfs:
            try:
                expanded, meta = _expand_spec_to_pdfs(root, str(args.pdfs))
            except Exception as e:
                print(f"error: invalid --pdfs spec: {e}", file=sys.stderr)
                return 2
            mdir = meta.get("dir")
            if isinstance(mdir, str) and mdir.strip():
                d = _resolve(root, mdir.strip())
                if d.exists() and d.is_dir():
                    active_pdf_dir = d.resolve()
            if active_pdf_dir is None and expanded:
                active_pdf_dir = expanded[0].resolve().parent
        else:
            # Fallback to alias Q if present, then repo-local PDFs if exists.
            alias_q = (os.getenv("PDFCATCH_ALIAS_Q") or "").strip()
            if alias_q:
                p = _resolve(root, alias_q)
                if p.exists() and p.is_dir():
                    active_pdf_dir = p.resolve()
            if active_pdf_dir is None:
                p = (root / "PDFs").resolve()
                if p.exists() and p.is_dir():
                    active_pdf_dir = p

        if active_pdf_dir is None:
            if args.cmd == "studio":
                print("info: no initial PDF directory inferred; open Studio and choose one in the sidebar.")
            else:
                print("error: could not infer PDF directory. Provide --pdfs or --dir.", file=sys.stderr)
                return 2

        extracted_dir: Path | None = None
        if args.save_files:
            extracted_dir = _resolve(root, str(args.save_files))
            extracted_dir.mkdir(parents=True, exist_ok=True)

        if args.cmd == "studio":
            section_norm = "Deteccao"
        else:
            section_map = {
                "Templates": "Deteccao",
                "Confirmacao": "Confirmacao",
                "Revisao": "Confirmacao",
                "Extracao": "Estatisticas",
                "Classificacao": "Estatisticas",
                "Indexacao": "Studio",
            }
            section_raw = str(getattr(args, "section", "Studio"))
            section_norm = section_map.get(section_raw, section_raw)

        print(f"Abrindo Web UI... section={section_norm} pdf_dir={active_pdf_dir}")
        rc_web = _launch_webui(
            root=root,
            active_pdf_dir=active_pdf_dir,
            extracted_dir=extracted_dir,
            section=str(section_norm),
            limit=max(0, int(getattr(args, "limit", 0) or 0)),
        )
        if rc_web != 0:
            print("error: failed to open Web UI", file=sys.stderr)
            return int(rc_web)
        return 0

    if args.cmd == "tui":
        tui_argv: list[str] = []
        if getattr(args, "candidate_pdf", None):
            tui_argv.append(str(args.candidate_pdf))
        if getattr(args, "templates", None):
            tui_argv += ["--templates", str(args.templates)]
        if getattr(args, "top_k", None) is not None:
            tui_argv += ["--top-k", str(int(args.top_k))]
        if bool(getattr(args, "no_stream", False)):
            tui_argv.append("--no-stream")
        if bool(getattr(args, "no_text", False)):
            tui_argv.append("--no-text")
        if bool(getattr(args, "no_image", False)):
            tui_argv.append("--no-image")
        if getattr(args, "image_dpi", None) is not None:
            tui_argv += ["--image-dpi", str(int(args.image_dpi))]
        if getattr(args, "image_hash_size", None) is not None:
            tui_argv += ["--image-hash-size", str(int(args.image_hash_size))]
        if getattr(args, "stream_prefilter_min", None) is not None:
            tui_argv += ["--stream-prefilter-min", str(float(args.stream_prefilter_min))]
        if getattr(args, "out", None):
            tui_argv += ["--out", str(args.out)]
        return _launch_tui(root=root, argv=tui_argv)

    if args.cmd in {"doc", "document"}:
        save_dir: Path | None = None
        if args.save_files:
            save_dir = _resolve(root, str(args.save_files))
            save_dir.mkdir(parents=True, exist_ok=True)
        elif bool(args.web):
            # Web flow expects exported files available for immediate visual validation.
            save_dir = _resolve(root, "docs_extracted")
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"info: --web sem --save-files; usando padrao: {save_dir}")

        templates_path = str(args.templates or os.getenv("PDFCATCH_TEMPLATES_JSON") or "configs/templates.json")
        templates_json = _resolve(root, templates_path)
        if not templates_json.exists():
            print(f"error: templates file not found: {templates_json}", file=sys.stderr)
            return 2
        templates = _load_templates(root, templates_json)

        despacho_specs: list[str] = list(args.despacho or [])
        cert_specs: list[str] = list(args.certidao or [])

        def _expand_specs_unique(specs: list[str]) -> list[Path]:
            pdfs: list[Path] = []
            seen: set[str] = set()
            for spec in specs:
                expanded, _meta = _expand_spec_to_pdfs(root, spec)
                for pth in expanded:
                    k = str(pth.resolve())
                    if k in seen:
                        continue
                    seen.add(k)
                    pdfs.append(pth)
            return pdfs

        if args.dry_run:
            for lab, specs in [("DESPACHO", despacho_specs), ("CERTIDAO_CM", cert_specs)]:
                if not specs:
                    continue
                uniq = _expand_specs_unique(specs)
                print(f"{lab}: {len(uniq)} PDFs")
                for i, pth in enumerate(uniq, start=1):
                    print(f"  {i:>3}. {pth}")
            return 0

        if not despacho_specs and not cert_specs:
            print("error: provide at least one of: -d/--despacho, -c/--certidao", file=sys.stderr)
            return 2
        if despacho_specs and cert_specs:
            print("error: use only one label per run: choose --despacho OR --certidao", file=sys.stderr)
            return 2

        active_specs = despacho_specs if despacho_specs else cert_specs
        web_active_pdf_dir = _infer_primary_pdf_dir(root, active_specs)

        if args.all and args.method:
            console.print("[red]error:[/red] use either --all (all methods) or -m/--method (selected methods), not both.")
            return 2

        top_n = int(args.top_n) if args.top_n is not None else int(os.getenv("PDFCATCH_TOP_N", "5") or "5")
        if top_n <= 0:
            console.print("[red]error:[/red] --top-n deve ser >= 1")
            return 2
        if args.min_p1 is not None and not (0.0 <= float(args.min_p1) <= 1.0):
            console.print("[red]error:[/red] --min-p1 deve estar entre 0 e 1")
            return 2
        if args.min_p2 is not None and not (0.0 <= float(args.min_p2) <= 1.0):
            console.print("[red]error:[/red] --min-p2 deve estar entre 0 e 1")
            return 2
        bytes_ref_mode = str(
            args.bytes_ref_mode
            or os.getenv("PDFCATCH_BYTES_REF_MODE")
            or "robust"
        ).strip().lower()
        if bytes_ref_mode not in {"robust", "legacy", "kmeans", "mean2"}:
            console.print("[red]error:[/red] --bytes-ref-mode deve ser robust, legacy, kmeans ou mean2")
            return 2

        # Normalize method list.
        raw_methods: list[str] = []
        for chunk in (args.method or []):
            if isinstance(chunk, list):
                raw_methods.extend([str(x) for x in chunk])
            else:
                raw_methods.append(str(chunk))
        methods: list[str]
        if args.all:
            methods = ["bytes", "tlsh", "stream", "text", "image"]
        elif raw_methods:
            norm: list[str] = []
            for m in raw_methods:
                v = str(m).strip().lower()
                if v in {"byte", "bytes", "size", "size_only"}:
                    v = "bytes"
                elif v in {"hash", "tlsh"}:
                    v = "tlsh"
                elif v in {"stream", "contents"}:
                    v = "stream"
                elif v in {"text", "txt"}:
                    v = "text"
                elif v in {"image", "img", "pixels", "dhash"}:
                    v = "image"
                elif v == "all":
                    v = "all"
                else:
                    console.print(f"[red]error:[/red] unknown method: {m!r}")
                    return 2
                if v not in norm:
                    norm.append(v)
            if "all" in norm:
                if len(norm) > 1:
                    console.print("[red]error:[/red] --method all nao pode ser combinado com outros metodos.")
                    return 2
                methods = list(_ALL_METHODS)
            else:
                methods = norm
        else:
            methods = ["bytes"]

        selected_methods = list(methods)
        kmeans_ref_by_label: dict[str, dict[str, Any]] = {}
        mean2_ref_by_label: dict[str, dict[str, Any]] = {}
        if bytes_ref_mode == "kmeans" and "bytes" in selected_methods:
            for _lab, _specs in [("DESPACHO", despacho_specs), ("CERTIDAO_CM", cert_specs)]:
                if not _specs:
                    continue
                _tpls = [t for t in templates if t.label == _lab]
                if not _tpls:
                    console.print(f"[red]error:[/red] no templates for label={_lab} (required by kmeans mode)")
                    return 2
                _tpls_seq = sorted(_tpls, key=lambda t: int(t.page_index))
                _slot_templates = _group_templates_by_slot(_tpls_seq)
                _train_pdfs = _expand_specs_unique(_specs)
                if not _train_pdfs:
                    console.print(f"[red]error:[/red] no train PDFs for label={_lab} (required by kmeans mode)")
                    return 2
                try:
                    _km_ref = _build_kmeans_bytes_reference(
                        root=root,
                        slot_templates=_slot_templates,
                        label=_lab,
                        train_pdfs=_train_pdfs,
                    )
                except Exception as exc:
                    console.print(f"[red]error:[/red] kmeans reference build failed for {_lab}: {exc}")
                    return 2
                kmeans_ref_by_label[_lab] = _km_ref
                _sm = _km_ref.get("summary") if isinstance(_km_ref.get("summary"), dict) else {}
                console.print(
                    "kmeans_ref "
                    f"label={_lab} target_cluster={_sm.get('target_cluster')} "
                    f"train_pdfs={_sm.get('train_pdf_count')} windows={_sm.get('train_window_count')} "
                    f"tips={_sm.get('tip_term_count')}",
                    style="bright_black",
                )
        if bytes_ref_mode == "mean2" and "bytes" in selected_methods:
            train_specs_cli = [str(x).strip() for x in (getattr(args, "bytes_train_spec", []) or []) if str(x).strip()]
            train_specs_env = _split_env_terms(str(os.getenv("PDFCATCH_BYTES_TRAIN_SPEC") or ""))
            for _lab, _specs in [("DESPACHO", despacho_specs), ("CERTIDAO_CM", cert_specs)]:
                if not _specs:
                    continue
                _tpls = [t for t in templates if t.label == _lab]
                if not _tpls:
                    console.print(f"[red]error:[/red] no templates for label={_lab} (required to resolve slot count)")
                    return 2
                _tpls_seq = sorted(_tpls, key=lambda t: int(t.page_index))
                _slot_templates = _group_templates_by_slot(_tpls_seq)
                _train_specs = list(train_specs_cli or train_specs_env or _specs)
                _train_pdfs = _expand_specs_unique(_train_specs)
                if not _train_pdfs:
                    console.print(
                        f"[red]error:[/red] no train PDFs for label={_lab} in mean2 mode "
                        f"(train_specs={_train_specs})"
                    )
                    return 2
                try:
                    _m2_ref = _build_mean2_bytes_reference(
                        slot_templates=_slot_templates,
                        label=_lab,
                        train_pdfs=_train_pdfs,
                    )
                except Exception as exc:
                    console.print(f"[red]error:[/red] mean2 reference build failed for {_lab}: {exc}")
                    return 2
                mean2_ref_by_label[_lab] = _m2_ref
                _sm2 = _m2_ref.get("summary") if isinstance(_m2_ref.get("summary"), dict) else {}
                console.print(
                    "mean2_ref "
                    f"label={_lab} train_pdfs={_sm2.get('train_pdf_count')} "
                    f"raw={_sm2.get('train_doc_count_used_raw')} kept={_sm2.get('train_doc_count_used_kept')} "
                    f"means={_sm2.get('mean_sizes')}",
                    style="bright_black",
                )
        expected_total_pdfs = 0
        for _lab, _specs in [("DESPACHO", despacho_specs), ("CERTIDAO_CM", cert_specs)]:
            if not _specs:
                continue
            expected_total_pdfs += len(_expand_specs_unique(_specs))
        processed_total = 0
        all_results: list[dict[str, Any]] = []
        process_summaries: list[dict[str, Any]] = []
        live_enabled = bool(sys.stdout.isatty())
        dashboard: LiveRunDashboard | None = None
        if live_enabled:
            dashboard = LiveRunDashboard(console=console, expected_rows=expected_total_pdfs)
            dashboard.start()

        def _emit_line(text: str = "", style: str | None = None) -> None:
            if dashboard is not None:
                dashboard.log(text, style)
            else:
                if style:
                    console.print(text, style=style)
                else:
                    print(text)

        try:
            if expected_total_pdfs > 0:
                _emit_line(f"PROGRESSO GERAL { _progress_bar(0, expected_total_pdfs) }", "bright_black")
                _emit_line("")
            for label, specs in [("DESPACHO", despacho_specs), ("CERTIDAO_CM", cert_specs)]:
                if not specs:
                    continue
                pdfs = _expand_specs_unique(specs)
                total_pdfs = len(pdfs)
                if total_pdfs <= 0:
                    continue

                for i, pdf_path in enumerate(pdfs, start=1):
                    process_id = _process_id_from_pdf_name(pdf_path)
                    _emit_line(
                        f"INICIANDO arquivo={pdf_path.name} processo={process_id} "
                        f"label={label} ({i}/{total_pdfs})",
                        "bright_black",
                    )

                    rows_for_process: list[dict[str, Any]] = []
                    for m in selected_methods:
                        if m == "tlsh" and _tlsh is None:
                            _emit_line("aviso: tlsh OFF (instale: pip install py-tlsh)", "#b87838")

                        rows = _run_doc_extract(
                            root=root,
                            templates=templates,
                            label=label,
                            method=m,
                            specs=[str(pdf_path)],
                            save_dir=None,
                            top_n=top_n,
                            min_p1=(float(args.min_p1) if args.min_p1 is not None else None),
                            min_p2=(float(args.min_p2) if args.min_p2 is not None else None),
                            bytes_ref_mode=bytes_ref_mode,
                            kmeans_reference=(
                                kmeans_ref_by_label.get(label)
                                if (bytes_ref_mode == "kmeans" and m == "bytes")
                                else None
                            ),
                            mean2_reference=(
                                mean2_ref_by_label.get(label)
                                if (bytes_ref_mode == "mean2" and m == "bytes")
                                else None
                            ),
                            emit_details=True,
                            candidate_position=(i, total_pdfs),
                            console=console,
                            log_sink=(dashboard.log if dashboard is not None else None),
                        )
                        rows_for_process.extend(rows)
                        all_results.extend(rows)
                        _emit_line("")

                    # Final strictly for this process (no mixing with other PDFs).
                    methods_map: dict[str, dict[str, Any]] = {}
                    rows_by_method: dict[str, dict[str, Any]] = {}
                    window_scores_by_method: dict[str, dict[tuple[int, ...], float]] = {}
                    for r in rows_for_process:
                        meth = str(r.get("method") or "")
                        if not meth:
                            continue
                        rows_by_method[meth] = r
                        if r.get("status") == "ok":
                            methods_map[meth] = {
                                "status": "ok",
                                "score": float(r.get("method_score") or 0.0),
                                "best_window": r.get("best_window"),
                                "model_reference": r.get("model_reference"),
                            }
                            ws: dict[tuple[int, ...], float] = {}
                            all_windows = r.get("all_windows")
                            if isinstance(all_windows, list):
                                for w in all_windows:
                                    if not isinstance(w, dict):
                                        continue
                                    tsc = w.get("total")
                                    pidx = _page_indices_from_window(w)
                                    if pidx and isinstance(tsc, (int, float)):
                                        ws[tuple(pidx)] = float(tsc)
                            if not ws:
                                bw = r.get("best_window")
                                if isinstance(bw, dict):
                                    tsc = r.get("method_score")
                                    pidx = _page_indices_from_window(bw)
                                    if pidx and isinstance(tsc, (int, float)):
                                        ws[tuple(pidx)] = float(tsc)
                            window_scores_by_method[meth] = ws
                        else:
                            methods_map[meth] = {
                                "status": "skipped",
                                "score": None,
                                "reason": r.get("reason"),
                                "model_reference": r.get("model_reference"),
                            }
                            window_scores_by_method[meth] = {}

                    valid_methods = [m for m in selected_methods if (methods_map.get(m) or {}).get("status") == "ok"]
                    all_keys: set[tuple[int, ...]] = set()
                    for m in valid_methods:
                        all_keys.update(window_scores_by_method.get(m, {}).keys())

                    best_pages: tuple[int, ...] | None = None
                    best_cov = -1
                    best_avg = -1.0
                    best_scores_by_method: dict[str, float] = {}
                    for key in sorted(all_keys):
                        scores_for_key: dict[str, float] = {}
                        for m in valid_methods:
                            sc = window_scores_by_method.get(m, {}).get(key)
                            if sc is not None:
                                scores_for_key[m] = float(sc)
                        cov = len(scores_for_key)
                        if cov <= 0:
                            continue
                        avg = float(sum(scores_for_key.values()) / float(cov))
                        if (cov > best_cov) or (cov == best_cov and avg > best_avg):
                            best_cov = cov
                            best_avg = avg
                            best_pages = key
                            best_scores_by_method = scores_for_key

                    parts: list[str] = []
                    for m in selected_methods:
                        sc = best_scores_by_method.get(m)
                        if sc is None:
                            parts.append(f"{m}=-")
                        else:
                            parts.append(f"{m}={float(sc):.4f}")
                    method_pages_parts: list[str] = []
                    for m in selected_methods:
                        md = methods_map.get(m) if isinstance(methods_map, dict) else None
                        bw = md.get("best_window") if isinstance(md, dict) else None
                        midx, mpages = _window_to_idx_pages(bw)
                        if midx == "-":
                            method_pages_parts.append(f"{m}=-")
                        else:
                            method_pages_parts.append(f"{m}=idx{midx}/p{mpages}")

                    final_score = float(best_avg) if best_pages is not None else None
                    exported_final_pdf: str | None = None
                    best_window: dict[str, Any] | None = None
                    idx0_txt = "-"
                    if best_pages is not None:
                        bpages = [int(v) for v in best_pages]
                        bs, be = int(min(bpages)), int(max(bpages))
                        idx0_txt = _idx0_compact(bpages)
                        best_window = {
                            "start_page_index": bs,
                            "end_page_index": be,
                            "start_page": bs + 1,
                            "end_page": be + 1,
                            "page_indices": bpages,
                        }

                    # Export exactly one document per candidate file (the best page index).
                    if save_dir is not None and best_pages is not None:
                        bpages = [int(v) for v in best_pages]
                        if bpages:
                            sidx, eidx = int(min(bpages)), int(max(bpages))
                            out_label_dir = (save_dir / label).resolve()
                            out_label_dir.mkdir(parents=True, exist_ok=True)
                            cand_path = Path(str(pdf_path)).resolve()
                            sha = _sha256_file(cand_path)
                            ts_out = datetime.now().strftime("%Y%m%d_%H%M%S")
                            idx_tok = _idx0_compact(bpages)
                            pag_tok = _pages_compact(bpages)
                            out_name = (
                                f"{_safe_stem(cand_path.stem)}__{sha[:12]}__{label}__FINAL"
                                f"__idx{idx_tok}__pag{pag_tok}__{ts_out}.pdf"
                            )
                            out_pdf = (out_label_dir / out_name).resolve()
                            cand_doc = fitz.open(str(cand_path))
                            try:
                                _export_pages_pdf(cand_doc, page_indices=bpages, out_path=out_pdf)
                            finally:
                                try:
                                    cand_doc.close()
                                except Exception:
                                    pass
                            exported_final_pdf = str(out_pdf)

                    def _methods_colored(items: list[str]) -> str:
                        out: list[str] = []
                        for mm in items:
                            c = _METHOD_COLORS.get(str(mm).lower(), "white")
                            out.append(f"[{c}]{mm}[/{c}]")
                        return ", ".join(out)

                    def _scores_colored(items: list[str]) -> str:
                        out: list[str] = []
                        for it in items:
                            k, v = (it.split("=", 1) + [""])[:2] if "=" in it else (it, "")
                            c = _METHOD_COLORS.get(str(k).lower(), "white")
                            if "=" in it:
                                out.append(f"[{c}]{k}[/{c}]={v}")
                            else:
                                out.append(f"[{c}]{it}[/{c}]")
                        return ", ".join(out)

                    def _method_pages_colored(items: list[str]) -> str:
                        out: list[str] = []
                        for it in items:
                            k, v = (it.split("=", 1) + [""])[:2] if "=" in it else (it, "")
                            c = _METHOD_COLORS.get(str(k).lower(), "white")
                            if "=" in it:
                                out.append(f"[{c}]{k}[/{c}]={v}")
                            else:
                                out.append(f"[{c}]{it}[/{c}]")
                        return ", ".join(out)

                    def _kv_line(key: str, value: str, *, value_style: str = "white", raw_markup: bool = False) -> None:
                        if raw_markup:
                            _emit_line(f"{key:<8} = {value}", "bright_black")
                        else:
                            _emit_line(f"{key:<8} = {value}", value_style)

                    # In live mode, keep final results in the top summary only.
                    # Bottom area should remain processing logs.
                    if dashboard is None:
                        _emit_line("Resultado final da etapa", "bold white")
                        _kv_line("PROCESSO", str(process_id))
                        _kv_line("ARQUIVO", str(pdf_path.name))
                        _kv_line("LABEL", str(label))
                        _kv_line("METODOS", _methods_colored(selected_methods), raw_markup=True)
                        _emit_line("DET_MET  = metodo -> idx0/pages | score", "bright_black")
                        for m in selected_methods:
                            md = methods_map.get(m) if isinstance(methods_map, dict) else None
                            bw = md.get("best_window") if isinstance(md, dict) else None
                            midx, mpages = _window_to_idx_pages(bw)
                            mscore = md.get("score") if isinstance(md, dict) else None
                            if midx == "-" or not isinstance(mscore, (int, float)):
                                line = f"{m:<8} -> -"
                            else:
                                line = f"{m:<8} -> idx{midx}/p{mpages} | {float(mscore):.4f}"
                            _emit_line(line, "bright_black")
                        if final_score is None:
                            _kv_line("FINAL_SC", "-")
                            _kv_line("BEST_IDX", "-")
                        else:
                            _kv_line("FINAL_SC", f"{final_score:.4f} ({final_score*100.0:0.2f}%)", value_style="bold white")
                            _kv_line("BEST_IDX", idx0_txt)
                        if exported_final_pdf:
                            _kv_line("EXPORTED", str(exported_final_pdf), value_style="bright_black")
                        _emit_line("")

                    process_summaries.append(
                        {
                            "label": label,
                            "process_id": process_id,
                            "candidate_pdf": str(pdf_path),
                            "selected_methods": selected_methods,
                            "bytes_ref_mode": bytes_ref_mode,
                            "methods": methods_map,
                            "final_score": final_score,
                            "best_window": best_window,
                            "scores_line": ", ".join(parts),
                            "scores_line_colored": _scores_colored(parts),
                            "method_pages_line": ", ".join(method_pages_parts),
                            "method_pages_line_colored": _method_pages_colored(method_pages_parts),
                            "best_idx0": idx0_txt,
                            "exported_pdf_final": exported_final_pdf,
                        }
                    )
                    if dashboard is not None:
                        dashboard.set_summaries(process_summaries)
                    processed_total += 1
                    _emit_line(f"PROGRESSO GERAL { _progress_bar(processed_total, expected_total_pdfs) }", "bright_black")
                    _emit_line("")

            if process_summaries and dashboard is None:
                _print_consolidated_report(console, process_summaries)
        finally:
            if dashboard is not None:
                dashboard.stop()

        if save_dir is not None:
            # Run-level index.
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            idx = {
                "run_ts": ts,
                "save_dir": str(save_dir),
                "selected_methods": selected_methods,
                "kmeans_reference_by_label": {
                    str(k): (v.get("summary") if isinstance(v, dict) else None)
                    for k, v in kmeans_ref_by_label.items()
                },
                "mean2_reference_by_label": {
                    str(k): (v.get("summary") if isinstance(v, dict) else None)
                    for k, v in mean2_ref_by_label.items()
                },
                "outputs": all_results,
                "process_summaries": process_summaries,
            }
            (save_dir / f"index__{ts}.json").write_text(
                json.dumps(idx, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"wrote: {save_dir / f'index__{ts}.json'}")

        if bool(getattr(args, "return_mode", False)):
            ts_ret = datetime.now().strftime("%Y%m%d_%H%M%S")
            io_dir = _resolve(root, str(getattr(args, "io_dir", None) or "io/out"))
            io_dir.mkdir(parents=True, exist_ok=True)
            detected_compact: list[dict[str, Any]] = []
            for ps in process_summaries:
                if not isinstance(ps, dict):
                    continue
                bw = ps.get("best_window") if isinstance(ps.get("best_window"), dict) else {}
                page_indices = bw.get("page_indices") if isinstance(bw, dict) else None
                if not isinstance(page_indices, list):
                    page_indices = []
                meth_out: dict[str, Any] = {}
                methods_map = ps.get("methods")
                if isinstance(methods_map, dict):
                    for mk, mv in methods_map.items():
                        if not isinstance(mv, dict):
                            continue
                        mbw = mv.get("best_window") if isinstance(mv.get("best_window"), dict) else {}
                        mpidx = mbw.get("page_indices") if isinstance(mbw, dict) else None
                        if not isinstance(mpidx, list):
                            mpidx = []
                        meth_out[str(mk)] = {
                            "status": mv.get("status"),
                            "score": mv.get("score"),
                            "page_indices": mpidx,
                            "model_reference": mv.get("model_reference"),
                        }
                detected_compact.append(
                    {
                        "process_id": ps.get("process_id"),
                        "label": ps.get("label"),
                        "candidate_pdf": ps.get("candidate_pdf"),
                        "exported_pdf_final": ps.get("exported_pdf_final"),
                        "final_score": ps.get("final_score"),
                        "best_idx0": ps.get("best_idx0"),
                        "best_page_indices": page_indices,
                        "methods": meth_out,
                    }
                )
            ret_payload = {
                "schema": "pdfcatch.detector.return.v1",
                "run_ts": ts_ret,
                "command": "doc",
                "cwd": str(root),
                "templates_json": str(templates_json),
                "bytes_ref_mode": bytes_ref_mode,
                "selected_methods": selected_methods,
                "top_n": int(top_n),
                "kmeans_reference_by_label": {
                    str(k): (v.get("summary") if isinstance(v, dict) else None)
                    for k, v in kmeans_ref_by_label.items()
                },
                "mean2_reference_by_label": {
                    str(k): (v.get("summary") if isinstance(v, dict) else None)
                    for k, v in mean2_ref_by_label.items()
                },
                "input_specs": {
                    "despacho": despacho_specs,
                    "certidao": cert_specs,
                },
                "detected": detected_compact,
                "process_summaries": process_summaries,
                "outputs": all_results,
            }
            ret_path = (io_dir / f"detector_return__{ts_ret}.json").resolve()
            ret_path.write_text(json.dumps(ret_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"return_json: {ret_path}")

        if bool(args.web):
            print("Abrindo Web UI...")
            rc_web = _launch_webui(
                root=root,
                active_pdf_dir=web_active_pdf_dir,
                extracted_dir=save_dir,
                section="Extracao",
                limit=0,
            )
            if rc_web != 0:
                print("error: failed to open Web UI", file=sys.stderr)
                return int(rc_web)
        return 0

    print("error: unknown command", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
