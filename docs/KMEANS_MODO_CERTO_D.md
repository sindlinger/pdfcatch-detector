# KMEANS Modo Certo (:D) - Deteccao Apenas

## Objetivo
Fixar um modo unico e reproduzivel para **deteccao** (nao extracao) em `:D`,
sem confundir com pipeline do extrator.

## Escopo
- Repo: `/mnt/c/git/pdfcatch-detector`
- Tipo: deteccao de par contiguo de paginas por bytes
- Modo: `bytes + kmeans`
- Dataset usado: `:D` validado (300 PDFs)

## Comando Canonico
```bash
cd /mnt/c/git/pdfcatch-detector
mkdir -p io/out/kmeans_detect_d

PDFCATCH_ALIAS_D=/mnt/c/git/pdfcatch-detector/io/train_despachos_valid \
PDFCATCH_BYTES_TRAIN_SPEC=/mnt/c/git/pdfcatch-detector/io/train_despachos_valid \
PDFCATCH_BYTES_MIN_ANY=0 \
PYTHONPATH=src \
python -m pdfcatch.cli doc \
  -d :D \
  --top-n 1 \
  --min-p1 0 \
  --min-p2 0 \
  --return \
  --io-dir io/out/kmeans_detect_d
```

## Script Canonico (recomendado)
```bash
cd /mnt/c/git/pdfcatch-detector
tools/run_kmeans_detect_d.sh
```

## Resultado de Referencia (2026-02-21)
- Log: `io/out/kmeans_detect_d/run_kmeans_detect_d_20260221_231537.log`
- Return: `io/out/kmeans_detect_d/detector_return__20260221_231634.json`
- Total: `300`
- Detectados: `299`
- Nao detectados: `1`

### Motivo do unico nao detectado
Arquivo de **1 pagina** (ineligivel para detector de par contiguo p1-p2):
- `SEI_006223_66.2025.8.15_p047-047.pdf`

Conclusao pratica:
- Taxa total do lote: `299/300 = 99.67%`
- Taxa em elegiveis (>=2 paginas): `299/299 = 100%`

## Verificacao Rapida
```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("io/out/kmeans_detect_d/detector_return__20260221_231634.json")
obj = json.loads(p.read_text(encoding="utf-8"))
rows = obj.get("detected", [])
det = [
    r for r in rows
    if isinstance(r.get("best_page_indices"), list)
    and len(r["best_page_indices"]) == 2
    and r.get("final_score") is not None
]
print("total", len(rows))
print("detected", len(det))
print("skipped", len(rows) - len(det))
PY
```

## Separacao Obrigatoria (para nao confundir de novo)
1. Detector (este doc):
   - so `pdfcatch-detector`
   - so arquivo `detector_return__*.json`
2. Extrator:
   - roda separado, consumindo `best_page_indices` do detector
   - gera `extractor_return__*.json`
