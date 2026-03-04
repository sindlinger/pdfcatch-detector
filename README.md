# pdfcatch-detector

Pipeline único baseado em **bytes + K-Means** para registrar catálogo e buscar documentos usando **uma única página de referência (page_index=0)**.

## Instalação rápida

```bash
cd /mnt/c/git/pdfcatch-detector
python -m venv .venv
source .venv/bin/activate
pip install -e .[prefilter]
```

## Fluxo em 4 passos

1) **catalog_collect** – extrai features da página 0 e gera um CSV.
2) **catalog_register** – cria/atualiza (com autorização) o catálogo oficial de famílias da página 0 e registra a **TAG**.
3) **similarity_search** – usa uma TAG já catalogada para buscar semelhança em PDF único ou diretório.
4) **elbow** – calcula a inércia para vários valores de *k* (método do cotovelo).

## Objetivo de Cada Parte

- **`catalog_collect`**
Objetivo: transformar PDFs conhecidos em amostras numéricas da **página 0** (entrada da catalogação).
- **Fase 1: `family_catalog` via `catalog_register`/`catalog_build`**
Objetivo: registrar referências de famílias de documentos conhecidos.
Pergunta que essa fase responde: **"este documento pertence a qual família conhecida?"**
- **Fase 2: inferência separada via `similarity_search`**
Objetivo: aplicar o catálogo em diretório misto/quarentena, sem novo registro de referência.
- **`similarity_score` (núcleo)**
Objetivo: pontuar cada página contra todas as famílias catalogadas em três etapas:
1) estrutural por features (`distance`, `similarity_pct`, `feature_similarity_pct`, `feature_delta_z`);
2) lexical por tokenização (`token_similarity_pct`, `token_stage`);
3) lexical por IA (`ai_similarity_pct`, `ai_stage`) com GPU obrigatória.
- **`live`**
Objetivo: aplicar em lote e dar visão operacional (`mode`, `family_id`, `match`, `confidence`).

## Contrato de Campos no JSON (`catalog_register`)

- Topo do arquivo:
- `catalog_tag`: tag dona do catálogo.
- `document_registry_contract`: definição formal de objetivo por fase.
- `json_field_naming`: convenção oficial de nomes (`family_id`, `family_registry`, etc.).
- `classification_contract`: regras de classificação e confiança.
- `catalog_manifest`: assinatura e origem da coleta de referência.
- `output_policy`: política de gravação oficial (catálogo único por TAG em diretório oficial).
- Em `pages.0` quando `classification_mode = family_catalog`:
- `family_registry`: lista oficial das famílias da página 0.
- Cada entrada tem: `family_id`, `family_centroid_norm`, `family_dist_mean`, `family_dist_max`, `family_reference_examples`.

## Contrato Operacional Obrigatório

1) Primeiro **cataloga famílias** (`catalog_collect` + `catalog_register`).
2) Depois executa **inferência** (`similarity_search`) usando sempre o **catálogo da page 0**.
3) Na inferência em diretório misto, o sistema varre as páginas do PDF e compara cada uma contra o catálogo da page 0.
4) Regra de família: menor distância ao centróide da página.
5) Regra de confiança por família (`family_catalog`):
- **high**: `distance <= dist_mean`
- **medium**: `dist_mean < distance <= dist_max`
- **low_out_of_family**: `distance > dist_max` (candidato a nova família)

Sem catálogo da página 0, a classificação fica como `uncataloged_page`.

## Caixa de Inferência (diretório misto)

- A inferência é uma fase separada da catalogação.
- O `catalog_register` só cria catálogo de famílias.
- A inferência (`similarity_search`) aplica o catálogo em novos PDFs via **TAG já catalogada**.
- O `similarity_search` varre as páginas do PDF e compara com a referência de capa catalogada.
- Não é permitida nova criação de referência para TAG já catalogada.
- Atualização de TAG existente só é permitida com intervenção explícita via arquivo `atualizar catálogo.json`.

## Procedimento Organizado (sem travar e sem bagunça)

1) Rode `catalog_collect` e `catalog_register` com uma `--tag` nova.
2) O `catalog_register` grava o catálogo em diretório oficial e registra a TAG no registry oficial.
3) Se tentar registrar novamente a mesma TAG sem `atualizar catálogo.json`, o `catalog_register` falha por proteção.
4) Em runtime, use `similarity_search` com `--tag` para inferir.
5) Em atualização autorizada, o sistema arquiva automaticamente referência anterior, referência nova e evento completo de auditoria.

## Diretório Oficial de Catálogo

- Registry oficial (único): `io/catalog_official/catalog_registry.json`
- Catálogo atual por tag: `io/catalog_official/tags/<tag>/catalog.json`
- Histórico/auditoria por tag: `io/catalog_official/tags/<tag>/history/<evento>/`
- Auditoria global (ex.: migração): `io/catalog_official/audit/`

Qualquer tentativa de usar `--registry` fora do caminho oficial é bloqueada.

## Features extraídas (11 por página)

- `chars`, `spaces`, `whitespace`, `non_whitespace` — contagens de texto bruto.
- `token_count` — quantidade de tokens (vocabulário por palavras) da página.
- `fill_percent` — proporção de caracteres não vazios.
- `line_count`, `empty_lines` — estatísticas de linhas.
- `font_count` — quantas fontes diferentes aparecem.
- `width_mm`, `height_mm` — dimensões da página em milímetros.

## Comandos principais

Coleta para catálogo (gera CSV da página 0):

```bash
PYTHONPATH=src:. python -m pdfcatch.family.catalog_collect \
  --dir catalog_dir_source --out io/out/p0.csv --echo
```

Catalogação (catálogo de página única):

```bash
PYTHONPATH=src:. python -m pdfcatch.family.catalog_register \
  --csv io/out/p0.csv --k 3 --examples 5 --tag despacho
```

Gabarito IA puro (somente capas validadas de despacho, sem classe negativa):

```bash
PYTHONPATH=src:. python -m pdfcatch.family.ai_embed_covers \
  --dir /caminho/so_com_despachos_validados \
  --out io/out/despacho_cover_embeddings.npz
```

Validação interna do gabarito IA puro (capa vs embedding paradigma, sem comparar capa-vs-capa):

```bash
PYTHONPATH=src:. python -m pdfcatch.family.ai_paradigm_validate \
  --embeddings io/out/despacho_cover_embeddings.npz \
  --paradigm-out io/out/despacho_paradigm_embedding.npz \
  --report-out io/out/despacho_paradigm_validation.json
```

Atualização de catálogo existente (somente com autorização explícita):

```json
{
  "tag": "despacho",
  "autorizar_atualizacao": true,
  "motivo": "ajuste controlado de referência"
}
```

Salve como `atualizar catálogo.json` na raiz do projeto e rode novamente o `catalog_register`.
O arquivo é consumido e arquivado no histórico da TAG.

Inferência por TAG já catalogada (salva JSON de saída):

```bash
PYTHONPATH=src:. python -m pdfcatch.family.similarity_search \
  --pdf /caminho/arquivo_multidoc.pdf --tag despacho \
  --ai-model BAAI/bge-reranker-v2-m3
```

No JSON salvo da inferência, cada página retorna:
- `family_comparison`: comparação com todas as famílias catalogadas (A, B, C...).
- `distance`: distância para cada família.
- `similarity_pct`: percentual geral de semelhança da página com cada família.
- `token_similarity_pct`: percentual de semelhança lexical da página com cada família (etapa separada).
- `token_stage`: detalhes da etapa lexical (weighted_jaccard, cosine, termos compartilhados).
- `ai_similarity_pct`: percentual de semelhança lexical por IA (CrossEncoder em GPU).
- `ai_stage`: detalhes do estágio de IA (modelo, device, melhor candidato, score).
- `feature_similarity_pct`: percentual por feature (`chars`, `token_count`, `fill_percent`, etc.) para cada família.
- `feature_delta_z`: diferença normalizada por feature, para auditoria/descartes.

IA lexical oficial:
- GPU CUDA obrigatória.
- Fallback para CPU removido.
- Sem GPU, a execução retorna `ai_runtime_error`.
- Dependências mínimas: `torch` com CUDA + `sentence-transformers`.

Inferência em lote (diretório):

```bash
PYTHONPATH=src:. python -m pdfcatch.family.similarity_search \
  --dir /caminho/quarentena --tag despacho \
  --ai-model BAAI/bge-reranker-v2-m3
```

Pontuação direta (núcleo de inferência, útil para debug):

```bash
PYTHONPATH=src:. python -m pdfcatch.family.similarity_score \
  --pdf /caminho/arquivo_multidoc.pdf --tag despacho \
  --ai-model BAAI/bge-reranker-v2-m3
```

Método do cotovelo (ajudar a escolher *k*):

```bash
PYTHONPATH=src:. python -m pdfcatch.family.elbow \
  --csv io/out/p0.csv --k-min 1 --k-max 8
```

Verificação de sintaxe/imports (check rápido):

```bash
tools/verificar_sintaxe.exe
```

Wrapper conveniente (usa `similarity_search`):

```bash
tools/doc.exe --pdf catalog_dir_source/2018198741_p011-012.pdf --tag despacho
```

Dashboard ao vivo (tabela em cima, log embaixo) para uma pasta:

```bash
tools/doc.exe --dir /caminho/quarentena --tag despacho
```

## Estrutura relevante

- `src/pdfcatch/family/features.py` – cálculo das 11 features + normalização.
- `src/pdfcatch/family/catalog_registry.py` – registro de TAGs catalogadas.
- `src/pdfcatch/family/catalog_collect.py` – coleta de features e resumo estatístico.
- `src/pdfcatch/family/catalog_register.py` – comando de catalogação (nome claro).
- `src/pdfcatch/family/catalog_build.py` – implementação da catalogação.
- `src/pdfcatch/family/similarity_search.py` – inferência (PDF único ou diretório), com saída salva.
- `src/pdfcatch/family/similarity_score.py` – núcleo de pontuação por TAG, comparando páginas do PDF com catálogo da capa.
- `src/pdfcatch/family/elbow.py` – inércia para vários *k*.
- `src/pdfcatch/family/live.py` – dashboard ao vivo por diretório.
- `tools/doc.exe` – wrapper para o `similarity_search`.

Mais nada do pipeline antigo é utilizado.
