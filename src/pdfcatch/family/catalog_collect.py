from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from pdfcatch.family.features import FEATURE_FIELDS, extract_page_features

CATALOG_PAGE_INDEX = 0

def run(dir_path: Path, out: Path, echo: bool, quiet: bool) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: List[str] = []
    source_dir = str(dir_path.resolve())
    header = ["source_dir", "file", "page_index"] + FEATURE_FIELDS
    rows.append(",".join(header))

    stats_matrix: List[List[float]] = []
    files_seen = 0
    invalid_files = 0
    invalid_examples: List[str] = []
    pdfs = sorted(dir_path.glob("*.pdf"))
    for pdf in pdfs:
        try:
            feat, idx = extract_page_features(pdf, CATALOG_PAGE_INDEX)
        except Exception as exc:
            invalid_files += 1
            if len(invalid_examples) < 10:
                invalid_examples.append(f"{pdf.name}: {exc}")
            continue
        if not feat:
            continue
        line = [source_dir, pdf.name, str(idx)] + [f"{v:.6f}" for v in feat]
        rows.append(",".join(line))
        stats_matrix.append(feat)
        files_seen += 1

    out.write_text("\n".join(rows), encoding="utf-8")
    total_rows = len(rows) - 1
    print(f"catalog_collect: {total_rows} linhas -> {out}")
    if invalid_files:
        print(f"catalog_collect: ignorados {invalid_files} arquivos invalidos/não-PDF")
        for ex in invalid_examples:
            print(f"  - {ex}")
    if echo:
        for line in rows:
            print(line)
    if (not quiet) and stats_matrix:
        arr = np.array(stats_matrix, dtype=float)
        tbl = []
        tbl.append(f"{'feature':<20} {'min':>10} {'median':>10} {'mean':>10} {'max':>10}")
        for i, f in enumerate(FEATURE_FIELDS):
            col = arr[:, i]
            tbl.append(f"{f:<20} {col.min():>10.2f} {np.median(col):>10.2f} {col.mean():>10.2f} {col.max():>10.2f}")
        print("")
        print(f"PDFs: {files_seen} | páginas (page={CATALOG_PAGE_INDEX}): {len(stats_matrix)}")
        print("\n".join(tbl))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Catalog Collect: extrai features da página 0 e gera CSV")
    ap.add_argument("--dir", required=True, help="Diretório de PDFs para catalogação")
    ap.add_argument("--out", default="io/out/family_catalog_collect.csv", help="CSV de saída")
    ap.add_argument("--echo", action="store_true", help="Imprime as linhas no stdout além de gravar o CSV")
    ap.add_argument("--quiet", action="store_true", help="Não imprime resumo estatístico")
    args = ap.parse_args(argv)
    return run(Path(args.dir), Path(args.out), bool(args.echo), bool(args.quiet))


if __name__ == "__main__":
    raise SystemExit(main())
