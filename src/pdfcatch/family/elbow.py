from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

from pdfcatch.family.features import FEATURE_FIELDS, normalize_matrix

CATALOG_PAGE_INDEX = 0

def run(csv_path: Path, k_min: int, k_max: int) -> int:
    df = pd.read_csv(csv_path)
    sub = df[df["page_index"] == CATALOG_PAGE_INDEX]
    if sub.empty:
        print(f"nenhuma linha para page_index={CATALOG_PAGE_INDEX} em {csv_path}")
        return 1
    X = sub[FEATURE_FIELDS].to_numpy(dtype=float)
    mean, std, Xn = normalize_matrix(X.tolist())
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        km.fit(Xn)
        print(f"k={k} inertia={km.inertia_:.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Elbow: imprime inércia para k no intervalo (somente página 0)")
    ap.add_argument("--csv", required=True, help="CSV do catalog_collect")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    args = ap.parse_args(argv)
    return run(Path(args.csv), int(args.k_min), int(args.k_max))


if __name__ == "__main__":
    raise SystemExit(main())
