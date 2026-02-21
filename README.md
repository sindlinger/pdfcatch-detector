# pdfcatch-detector

Repositório isolado e encapsulado para **detecção de documentos PDF**.

Conteúdo incluído:

- motor de detecção por métodos: `bytes`, `tlsh`, `stream`, `text`, `image`;
- módulos em `Modules/detector/`;
- CLI em `src/pdfcatch/cli.py` (`cli doc ...`);
- modelos em `Models/`;
- templates em `configs/templates.json`.

## Estrutura

```text
pdfcatch-detector/
  Models/
  Modules/detector/
  Templates/
  configs/
    run.env
    run.env.example
    templates.json
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
cli doc -d :Q8-8 -m bytes --top-n 1
```

Exemplo com thresholds por execução:

```bash
cli doc -d :Q8-8 -m bytes --top-n 5 --min-p1 0.60 --min-p2 0.40
```

## Uso (batch)

```bash
python tools/batch_detect.py
```

## Configuração

Edite `configs/run.env` para:

- apontar o diretório dos PDFs de entrada;
- configurar alias (`PDFCATCH_ALIAS_Q`);
- ajustar métodos e limites de score.

