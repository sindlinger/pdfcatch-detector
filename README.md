# pdfcatch-detector

Repositório isolado e encapsulado para **detecção de documentos PDF**.

Conteúdo incluído:

- motor de detecção por métodos: `bytes`, `tlsh`, `stream`, `text`, `image`;
- módulos em `Modules/detector/`;
- módulo de extração textual por âncoras `before/after` em `Modules/extrator/`;
- CLI em `src/pdfcatch/cli.py` (`cli doc ...`);
- modelos em `Models/`;
- templates em `configs/templates.json`.

## Estrutura

```text
pdfcatch-detector/
  Models/
  Modules/detector/
  Modules/extrator/
  Templates/
  configs/
    run.env
    run.env.example
    templates.json
  io/
    in/
    out/
  src/pdfcatch/
    cli.py
    core/
    lib/
    main.py
  tools/
    batch_detect.py
```

## Setup

```bash
cd /mnt/c/git/pdfcatch-detector
python -m venv .venv
source .venv/bin/activate
pip install -e .[prefilter]
```

## Uso (CLI principal)

```bash
cli doc -d :Q8-8 --top-n 1
```

## Uso (extração textual por âncoras)

```bash
cli extract-text --pdf :Q8-8 --anchors configs/anchor_text_fields.example.json
```

Salvar JSON:

```bash
cli alt-extract --pdf :Q1-10 --anchors configs/anchor_text_fields.example.json --out outputs/anchor_text.json
```

Exemplo com thresholds por execução:

```bash
cli doc -d :Q8-8 --top-n 5 --min-p1 0.60 --min-p2 0.40
```

Retorno JSON para encadeamento entre módulos:

```bash
cli doc -d :Q8-8 --top-n 1 --return
```

Esse modo grava automaticamente:

- `io/out/detector_return__YYYYmmdd_HHMMSS.json`

Consumir esse retorno no extrator textual:

```bash
cli extract-text --from-return latest --anchors configs/anchor_text_fields.example.json
```

Retorno JSON do extrator:

```bash
cli extract-text --from-return latest --anchors configs/anchor_text_fields.example.json --return
```

Gera:

- `io/out/extractor_return__YYYYmmdd_HHMMSS.json`

## Uso (batch)

```bash
python tools/batch_detect.py
```

## Uso (batch validado: piloto + gate + full)

Fluxo recomendado para lote grande com validação manual antes de executar tudo:

```bash
python tools/validated_batch_detect.py pilot \
  --input-dir /caminho/do/lote \
  --label DESPACHO \
  --train-dir /caminho/treino_fixo
```

Depois de preencher `manual_correct` no CSV de revisão:

```bash
python tools/validated_batch_detect.py full \
  --run-dir outputs/validated_batch__<ts> \
  --review-csv outputs/validated_batch__<ts>/pilot/review_template.csv
```

Detalhes em `docs/VALIDATED_BATCH_FLOW.md`.

## Configuração

Edite `configs/run.env` para:

- apontar o diretório dos PDFs de entrada;
- configurar alias (`PDFCATCH_ALIAS_Q`);
- ajustar métodos e limites de score.
