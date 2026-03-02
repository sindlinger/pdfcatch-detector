# Demonstracao KMeans (Detector Bytes - Rota Unica)

## Objetivo
Padronizar a validação do detector de despacho com **apenas uma rota**:
- seleção por tamanho de página + `KMeans` para pontuação final.

## Lote
- Lote recomendado: `:Q` (todos os PDFs aplicáveis).
- Return de referência: `io/out_all_kmeans/detector_return__YYYYmmdd_HHMMSS.json`

## Como o fluxo roda hoje
1. Treino:
   - usa PDFs de treino fixos (definidos por `PDFCATCH_BYTES_TRAIN_SPEC`);
   - cria janelas contíguas de 2 páginas por PDF;
   - extrai features de bytes: `log_p1`, `log_p2`, `log_total`, `ratio_p2_p1`, `spread_max_min`, `share_p1`, `log_ratio_p1_p2`;
   - ajusta `KMeans` e seleciona cluster DESPACHO via âncoras textuais.
2. Inferência:
   - para cada candidato, calcula score por janela:
     - similaridade por `size_total`;
     - similaridade de cluster alvo (`kmeans_similarity`);
     - `final = w_size*size_total + w_cluster*kmeans_similarity`;
   - mantém top-1 por janela.

## Comando canônico de geração
```bash
cd /mnt/c/git/pdfcatch-detector
PYTHONPATH=src python -m pdfcatch.cli doc -d :Q \
  --top-n 1 --min-p1 0 --min-p2 0 --return --io-dir io/out_all_kmeans
```

## Observações de comparação
- Versões anteriores de validação fora deste padrão foram removidas.
- O contrato do detector continua no `detector_return__*.json` para o extrator via `best_page_indices`.
