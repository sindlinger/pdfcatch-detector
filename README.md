# pdfcatch-detector

API unificada com duas abordagens de IA:

- `catalogacao_classica` (catálogo + paradigma por similaridade)
- `qwen_supervisionado` (dataset supervisionado + treino GPU + inferência)

## Estrutura Atual

- Código público:
  - `src/pdfcatch/catalogacao_classica`
  - `src/pdfcatch/qwen_supervisionado`
- API:
  - `src/pdfcatch/api_server.py`
- Dados isolados do Qwen:
  - `io/qwen_isolated/input`
  - `io/qwen_isolated/catalogs`
  - `io/qwen_isolated/models`
  - `io/qwen_isolated/outputs`

## Instalação

```bash
cd /mnt/c/git/pdfcatch-detector
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Subir API

```bash
pdfcatch-api --host 0.0.0.0 --port 8000
```

ou

```bash
PYTHONPATH=src:. python -m pdfcatch.api_server --host 0.0.0.0 --port 8000
```

Healthcheck:

```bash
curl http://localhost:8000/health
```

## Endpoints

### Catalogação Clássica

- `POST /api/v1/catalogacao-classica/build-catalog`
- `POST /api/v1/catalogacao-classica/find-docs`

Exemplo:

```bash
curl -X POST http://localhost:8000/api/v1/catalogacao-classica/build-catalog \
  -H 'Content-Type: application/json' \
  -d '{
    "tag": "despacho",
    "dir": "/mnt/c/git/operpdf/reference/models/despachos",
    "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  }'
```

### Qwen Supervisionado

- `POST /api/v1/qwen-supervisionado/build-catalog`
- `POST /api/v1/qwen-supervisionado/find-docs`
- `POST /api/v1/qwen-supervisionado/prepare-dataset`
- `POST /api/v1/qwen-supervisionado/fit-model`
- `POST /api/v1/qwen-supervisionado/predict-model`

Exemplo (treino supervisionado):

```bash
curl -X POST http://localhost:8000/api/v1/qwen-supervisionado/fit-model \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_dir": "io/qwen_isolated/input/dataset_supervisionado_despacho_v1",
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "epochs": 20,
    "batch_size_embed": 8,
    "batch_size_train": 256,
    "max_text_chars": 3000,
    "max_seq_length": 384
  }'
```

Exemplo (inferência supervisionada):

```bash
curl -X POST http://localhost:8000/api/v1/qwen-supervisionado/predict-model \
  -H 'Content-Type: application/json' \
  -d '{
    "model_dir": "io/qwen_isolated/models/supervised_dataset_supervisionado_despacho_v1_20260305_061415",
    "dir": "io/qwen_isolated/input/quarentena",
    "out": "io/qwen_isolated/outputs/predict_quarentena_gpu.json"
  }'
```

## Observações

- A abordagem Qwen bloqueia caminhos fora de `io/qwen_isolated/*`.
- Treino e inferência supervisionados exigem GPU CUDA ativa.
- O diretório `lixeira/` contém histórico de arquivos removidos/movidos.
