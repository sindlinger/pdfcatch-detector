#!/usr/bin/env python
"""
Pré-treino por página (p1 ou p2):
- Extrai features numéricas e algumas categóricas de uma página específica de cada PDF em PDFs/despachos.
- Features: page_bytes (tamanho do PDF gerado só com a página), chars, fill_percent, font_count, top_font, width_mm, height_mm, images, qr_like, signature_present.
- Opcional: salva o PDF de 1 página reconstruído para inspeção.
- Salva CSV em io/out/train_page_features.csv e referência numérica em io/out/page_kmeans_ref_despacho.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import fitz
import numpy as np
import pdfcatch.cli as cli
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table


NUM_FEATURE_FIELDS = [
    "page_bytes",
    "chars",
    "fill_percent",
    "spaces",
    "whitespace",
    "non_whitespace",
    "line_count",
    "empty_lines",
    "font_count",
    "width_mm",
    "height_mm",
    "images",
    "qr_like",
    "signature_present",
]

CSV_FIELDS = NUM_FEATURE_FIELDS + ["top_font"]


def _font_info(page: fitz.Page) -> tuple[int, str]:
    try:
        fonts = [b.get("font", "") for b in page.get_text("dict").get("blocks", []) if b.get("type") == 0 for l in b.get("lines", []) for s in l.get("spans", [])]
    except Exception:
        fonts = []
    if not fonts:
        return 0, ""
    c = Counter(fonts)
    top = c.most_common(1)[0][0]
    return len(c), str(top)


def _signature_flag(page: fitz.Page) -> int:
    # Marca assinatura apenas se houver alguma imagem na faixa inferior da página (possível rubrica/assinatura).
    try:
        imgs = page.get_images(full=True) or []
        height = float(page.rect.height)
        for img in imgs:
            xref = img[0]
            rects = page.get_image_rects(xref)
            if any(r.y1 >= 0.7 * height for r in rects):
                return 1
    except Exception:
        return 0
    return 0


def extract_page_features(pdf_path: Path, page_index: int) -> tuple[list[float], str]:
    doc = fitz.open(pdf_path)
    try:
        if doc.page_count <= page_index:
            return [], ""
        page = doc.load_page(page_index)
        print(f"[trace] extraindo page_bytes -> {pdf_path.name} p{page_index+1}")
        try:
            page_bytes = len(cli._one_page_pdf_bytes(doc=doc, page_index=page_index))
        except Exception:
            page_bytes = 0
        print(f"[trace] page_bytes={page_bytes}")
        print(f"[trace] extraindo texto -> {pdf_path.name} p{page_index+1}")
        text = page.get_text("text") or ""
        total_chars = len(text)
        spaces = text.count(" ") if text else 0
        whitespace = sum(1 for ch in text if ch.isspace()) if text else 0
        non_whitespace = total_chars - whitespace
        fill_percent = float(non_whitespace) / float(total_chars) if total_chars else 0.0
        lines = text.splitlines()
        line_count = len(lines)
        empty_lines = sum(1 for l in lines if l.strip() == "")
        print(f"[trace] chars={total_chars} spaces={spaces} whitespace={whitespace} non_ws={non_whitespace} fill_percent={fill_percent:.3f} lines={line_count} empty_lines={empty_lines}")
        print(f"[trace] extraindo fontes -> {pdf_path.name} p{page_index+1}")
        font_count, top_font = _font_info(page)
        print(f"[trace] font_count={font_count} top_font={top_font}")
        print(f"[trace] extraindo imagens -> {pdf_path.name} p{page_index+1}")
        imgs = page.get_images(full=True) or []
        images = len(imgs)
        qr_like = 0
        for img in imgs:
            try:
                w, h = float(img[2]), float(img[3])
                if w > 50 and h > 50:
                    ratio = w / h
                    if 0.8 <= ratio <= 1.25:
                        qr_like = 1
                        break
            except Exception:
                continue
        print(f"[trace] images={images} qr_like={qr_like}")
        width_mm = float(page.rect.width) * 25.4 / 72.0
        height_mm = float(page.rect.height) * 25.4 / 72.0
        print(f"[trace] page size mm=({width_mm:.2f} x {height_mm:.2f})")
        print(f"[trace] detectando assinatura (imagem no rodapé) -> {pdf_path.name} p{page_index+1}")
        signature = _signature_flag(page)
        print(f"[trace] signature_present={signature}")
        feats = [
            float(page_bytes),
            float(total_chars),
            fill_percent,
            float(spaces),
            float(whitespace),
            float(non_whitespace),
            float(line_count),
            float(empty_lines),
            float(font_count),
            width_mm,
            height_mm,
            float(images),
            float(qr_like),
            float(signature),
        ]
        return feats, top_font
    finally:
        doc.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pagina", type=int, required=True, help="página 1-based a extrair de cada PDF em PDFs/despachos")
    parser.add_argument("--verbose", action="store_true", help="mostrar features da página de cada PDF")
    parser.add_argument("--salvar-pagina", action="store_true", help="gravar PDF de 1 página gerado para cada arquivo")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    full_dir = root / "PDFs" / "despachos"
    out_dir = root / "io" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = out_dir / "pages" / f"p{args.pagina:02d}"
    if args.salvar_pagina:
        dump_dir.mkdir(parents=True, exist_ok=True)

    console = Console()
    console.rule("[bold cyan]Pré-treino por página (p1 ou p2)")
    console.log(f"PDFs completos: {full_dir}")
    console.log(f"Página escolhida: {args.pagina}")
    if args.verbose:
        console.log("Verbose ON: será mostrado o vetor de cada PDF.")
    console.print()

    rows: list[list[str | float | int]] = []
    skipped: list[str] = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        pdfs = sorted(full_dir.glob("*.pdf"))
        task = progress.add_task(f"[yellow]Extraindo página {args.pagina}", total=len(pdfs))
        for pdf in pdfs:
            console.log(f"[dim]abrindo[/dim] {pdf.name}")
            feats, top_font = extract_page_features(pdf, max(0, args.pagina - 1))
            label_value: int | str = args.pagina
            feature_fields = CSV_FIELDS

            if feats:
                rows.append([f"{pdf.name}__p{args.pagina:02d}.pdf", label_value, *feats, top_font])
                if args.salvar_pagina:
                    try:
                        blob = cli._one_page_pdf_bytes(doc=fitz.open(pdf), page_index=max(0, args.pagina - 1))
                        out_file = dump_dir / f"{pdf.stem}__p{args.pagina:02d}.pdf"
                        out_file.write_bytes(blob)
                        console.log(f"[dim]salvo 1p[/dim] {out_file} ({len(blob)} bytes)")
                    except Exception as exc:
                        console.log(f"[red]falha ao salvar 1p[/red] {pdf.name}: {exc}")
                if args.verbose:
                    table_pdf = Table(title=f"{pdf.name} (página {args.pagina})", show_lines=False, box=None)
                    table_pdf.add_column("feature", justify="left")
                    table_pdf.add_column("valor", justify="right")
                    for fname, fval in zip(feature_fields, [*feats, top_font]):
                        if fname == "file_size" or fname.endswith("bytes"):
                            val_str = f"{fval:.0f}"
                        elif fname in ("width_mm", "height_mm"):
                            val_str = f"{fval:.2f}"
                        elif fname == "top_font":
                            val_str = str(fval)
                        else:
                            val_str = f"{fval:.3f}"
                        table_pdf.add_row(fname, val_str)
                    console.print(table_pdf)
            else:
                skipped.append(pdf.name)
            progress.advance(task)

    csv_path = out_dir / "train_page_features.csv"
    feature_fields_csv = list(CSV_FIELDS)
    fields = ["file", "label"] + feature_fields_csv
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)
    console.print(f"[green]CSV salvo[/green]: {csv_path} linhas={len(rows)}")

    # Centroides supervisionados
    feats = [r[2:2 + len(NUM_FEATURE_FIELDS)] for r in rows]
    if not feats:
        console.print("[red]Nenhum PDF encontrado para calcular o tamanho.[/red]")
        if skipped:
            console.print(f"[dim]Pulados:[/dim] {len(skipped)} arquivos")
        return 1
    X = np.array(feats, dtype=float)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std_adj = np.where(std <= 1e-9, 1.0, std)

    def cent(fs: list[list[float]]) -> list[float]:
        arr = np.array(fs, dtype=float)
        return [float(v) for v in ((arr - mean) / std_adj).mean(axis=0).tolist()]

    c_norm = cent(feats)
    c_raw = [float(x) for x in (np.array(c_norm) * std_adj + mean).tolist()]
    ref = {
        "label": "DESPACHO",
        "feature_fields": feature_fields_csv,
        "feature_mean": [float(v) for v in mean.tolist()],
        "feature_std": [float(v) for v in std.tolist()],
        "centroids_norm": [c_norm],
        "centroids": [c_raw],
        "cluster_page_map": {0: -1},
        "target_cluster_by_page": {"-1": 0},
        "sample_count": int(len(feats)),
    }
    ref_path = out_dir / "page_kmeans_ref_despacho.json"
    ref_path.write_text(json.dumps(ref, ensure_ascii=False, indent=2))
    console.print(f"[green]Referência salva[/green]: {ref_path} samples={ref['sample_count']}")

    cents = ref["centroids"]
    table = Table(title="Centróide (página escolhida)")
    table.add_column("Feature", justify="left", no_wrap=True)
    table.add_column("centroid", justify="right")
    for f, a in zip(ref["feature_fields"], cents[0]):
        table.add_row(f, f"{a:.4f}")
    console.print()
    console.print(table)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
