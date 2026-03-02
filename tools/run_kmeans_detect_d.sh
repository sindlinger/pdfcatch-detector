#!/usr/bin/env bash
set -euo pipefail

# Canonical detector-only run for :D using bytes+kmeans.
# Keeps detection mode isolated from extractor flow.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/io/out/kmeans_detect_d}"
TRAIN_DIR="${PDFCATCH_TRAIN_D_DIR:-$ROOT_DIR/io/train_despachos_valid}"

mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$OUT_DIR/run_kmeans_detect_d_${TS}.log"

(
  cd "$ROOT_DIR"
  PDFCATCH_ALIAS_D="$TRAIN_DIR" \
  PDFCATCH_BYTES_TRAIN_SPEC="$TRAIN_DIR" \
  PDFCATCH_BYTES_MIN_ANY=0 \
  PYTHONPATH=src \
  python -m pdfcatch.cli doc \
    -d :D \
    --top-n 1 \
    --min-p1 0 \
    --min-p2 0 \
    --return \
    --io-dir "$OUT_DIR"
) >"$LOG" 2>&1

echo "log: $LOG"
RET_JSON="$(grep -Eo 'return_json:[[:space:]]+[^[:space:]]+' "$LOG" | tail -n1 | awk '{print $2}' || true)"
echo "detector_return: ${RET_JSON:-<not-found>}"

if [[ -n "${RET_JSON:-}" && -f "$RET_JSON" ]]; then
  python - "$RET_JSON" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
obj = json.loads(p.read_text(encoding="utf-8"))
rows = obj.get("detected", [])
det = [
    r for r in rows
    if isinstance(r.get("best_page_indices"), list)
    and len(r["best_page_indices"]) == 2
    and r.get("final_score") is not None
]
skip = len(rows) - len(det)
print(f"total={len(rows)} detected={len(det)} skipped={skip}")
if skip:
    print("skipped_processes:")
    for r in rows:
        ok = isinstance(r.get("best_page_indices"), list) and len(r["best_page_indices"]) == 2 and r.get("final_score") is not None
        if not ok:
            print(f"- {r.get('process_id')} | {r.get('candidate_pdf')}")
PY
fi
