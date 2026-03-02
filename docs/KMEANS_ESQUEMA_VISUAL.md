# Esquema Visual - Detector KMeans (Bytes)

## Arquivos de referência do fluxo consolidado
- `io/out_all_kmeans/detector_return__*.json`
- retorno por `extract` deve consumir apenas `best_page_indices`.

## Pipeline consolidado (único)
```mermaid
flowchart LR
  subgraph T["TREINO (bytes)"]
    T1[PDFs de treino] --> T2[Janelas contíguas de 2 páginas]
    T2 --> T3[Features bytes: log_p1, log_p2, log_total, ratio_p2_p1, spread_max_min, share_p1, log_ratio_p1_p2]
    T3 --> T4[KMeans fit]
    T4 --> T5[Centroides]
    T6[Âncoras textuais] --> T7[Escolha do cluster DESPACHO]
    T5 --> T7 --> T8[Modelo canônico (cluster-alvo)]
  end

  subgraph I["INFERÊNCIA (por PDF)"]
    I1[PDF candidato] --> I2[Janelas contíguas]
    I2 --> I3[size_total + cluster_sim]
    I3 --> I4[final = w_size*size_total + w_cluster*cluster_sim]
    I4 --> I5[Melhor janela]
    I5 --> O1[detector_return.json]
  end

  T8 --> I4
```

## Saída consolidada esperada
- `best_window.page_indices`: somente as páginas candidatas finais.
- `best_window.kmeans_*`: metadados de similaridade e cluster.
- `model_reference.kmeans`: resumo do cluster-alvo e parâmetros de treino.

## Contrato operacional
- Detector e extrator continuam em etapas separadas.
- Neste repositório, a rota ativa de produção é:
  **bytes + kmeans**.
