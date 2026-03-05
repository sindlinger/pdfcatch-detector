from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from pdfcatch.ia.catalog_registry import DEFAULT_REGISTRY_PATH, normalize_tag, resolve_catalog_by_tag
from pdfcatch.ia.similarity_score import score_pdf_against_catalog


def _build_table(rows: list[dict]) -> Table:
    table = Table(box=None, expand=True, show_header=True, header_style="bold cyan")
    table.add_column("#", style="bright_black", width=3)
    table.add_column("arquivo", overflow="fold")
    table.add_column("src_pg", width=7)
    table.add_column("mode", width=16)
    table.add_column("family_id", width=9)
    table.add_column("match", overflow="fold")
    table.add_column("dist", width=8)
    table.add_column("conf", width=16)
    table.add_column("status", width=20)
    for i, r in enumerate(rows, start=1):
        table.add_row(
            str(i),
            r.get("file", ""),
            str(r.get("source_page_index")) if r.get("source_page_index") is not None else "-",
            str(r.get("mode") or "-"),
            str(r.get("family_id")) if r.get("family_id") is not None else "-",
            str(r.get("match") or "-"),
            f"{r.get('distance'):.4f}" if isinstance(r.get("distance"), (int, float)) else "-",
            str(r.get("confidence") or "-"),
            str(r.get("status") or "-"),
        )
    return table


def _render(rows: list[dict], logs: List[str]) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(Panel(_build_table(rows), title="progresso"), ratio=3),
        Layout(Panel("\n".join(logs[-15:]), title="log"), ratio=1),
    )
    return layout


def run(dir_path: Path, tag: str, registry_path: Path) -> int:
    console = Console()
    pdfs = sorted(p for p in dir_path.glob("*.pdf") if p.is_file())
    if not pdfs:
        console.print(f"[red]nenhum PDF em {dir_path}")
        return 1

    ref = None
    try:
        import json

        norm_tag = normalize_tag(tag)
        ref_path = resolve_catalog_by_tag(registry_path, norm_tag)
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - falha de IO
        console.print(f"[red]erro resolvendo tag/catalogo: {exc}")
        return 1

    rows: list[dict] = []
    logs: list[str] = []
    logs.append(f"tag={norm_tag} catalogo={ref_path}")

    with Live(_render(rows, logs), console=console, refresh_per_second=6, transient=False) as live:
        for pdf in pdfs:
            logs.append(f"processando {pdf.name}")
            try:
                # usa o núcleo de pontuação de similaridade do catálogo.
                batch_res = score_pdf_against_catalog(pdf, ref)
                best = batch_res.get("best_page") or {}
                best_match = (best.get("best_match") or {}).get("file")
                rows.append(
                    {
                        "file": pdf.name,
                        "source_page_index": best.get("page_index"),
                        "mode": best.get("mode"),
                        "family_id": best.get("family_id"),
                        "match": best_match,
                        "distance": best.get("distance"),
                        "confidence": best.get("confidence"),
                        "status": batch_res.get("status") or best.get("family_status"),
                    }
                )
            except Exception as exc:  # pragma: no cover - falha runtime
                rows.append(
                    {
                        "file": pdf.name,
                        "source_page_index": None,
                        "mode": "-",
                        "family_id": None,
                        "match": "-",
                        "distance": None,
                        "confidence": "-",
                        "status": "erro",
                    }
                )
                logs.append(f"erro {pdf.name}: {exc}")
            live.update(_render(rows, logs))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Dashboard live: similarity search por tag catalogada em diretório de PDFs"
    )
    ap.add_argument("--dir", required=True, help="Diretório com PDFs")
    ap.add_argument("--tag", required=True, help="TAG já catalogada")
    ap.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY_PATH),
        help="Registry oficial de tags catalogadas",
    )
    args = ap.parse_args(argv)
    return run(Path(args.dir), str(args.tag), Path(args.registry))


if __name__ == "__main__":
    raise SystemExit(main())
