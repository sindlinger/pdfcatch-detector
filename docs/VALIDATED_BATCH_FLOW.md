# Validated Batch Flow (Pilot + Gate + Full)

Este fluxo implementa a validacao em duas fases para evitar executar lote grande sem validar qualidade antes.

## 1) Pilot (amostra aleatoria)

- Le apenas PDFs da pasta raiz (`--input-dir`, sem recursao).
- Seleciona amostra aleatoria (`--sample-size`, `--sample-seed`).
- Roda detector (`cli doc`) com `Bytes+KMeans`.
- Gera CSV para revisao manual com colunas `manual_correct` e `manual_notes`.

Exemplo:

```bash
python tools/validated_batch_detect.py pilot \
  --input-dir /caminho/do/lote \
  --label DESPACHO \
  --train-dir /caminho/treino_fixo \
  --sample-size 20 \
  --sample-seed 42 \
  --min-correct 18
```

Saidas principais:

- `outputs/validated_batch__<ts>/run_manifest.json`
- `outputs/validated_batch__<ts>/pilot/review_template.csv`
- `outputs/validated_batch__<ts>/pilot/run.log`

## 2) Revisao manual

No arquivo `review_template.csv`, preencha `manual_correct` por linha com um valor booleano:

- verdadeiro: `1`, `true`, `yes`, `y`, `ok`, `correct`
- falso: `0`, `false`, `no`, `n`, `wrong`, `incorrect`

## 3) Full (somente se gate aprovado)

- Gate: `manual_correct >= min_correct` (default `18`).
- Se aprovado, roda o lote completo no `--input-dir` original.
- Se reprovado, interrompe o full run e registra `gate_report.json`.

Exemplo:

```bash
python tools/validated_batch_detect.py full \
  --run-dir outputs/validated_batch__<ts> \
  --review-csv outputs/validated_batch__<ts>/pilot/review_template.csv
```

Saidas principais do full:

- `outputs/validated_batch__<ts>/full/io/detector_return__*.json`
- `outputs/validated_batch__<ts>/full/pages_found/`
- `outputs/validated_batch__<ts>/full/skipped_rows.csv`
- `outputs/validated_batch__<ts>/final_report.json`
