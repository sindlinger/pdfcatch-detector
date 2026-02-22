# Esquema Visual - Detector KMeans (Bytes)

## Referencias usadas nesta visualizacao
- Comparacao robust x kmeans (texto):
  - `/mnt/c/git/pdfcatch/io/out/compare_q_all_by_pdf_robust_vs_kmeans__20260221_1700.txt`
- FP proxy (texto):
  - `/mnt/c/git/pdfcatch/io/out/fp_proxy_q_all_robust_vs_kmeans__20260221_1704.txt`
- Returns do detector:
  - `/mnt/c/git/pdfcatch-detector/io/out_all_robust/detector_return__20260221_164107.json`
  - `/mnt/c/git/pdfcatch-detector/io/out_all_kmeans/detector_return__20260221_165542.json`
- Returns do extrator:
  - `/mnt/c/git/pdfcatch-extractor/io_all_robust/out/extractor_return__20260221_165652.json`
  - `/mnt/c/git/pdfcatch-extractor/io_all_kmeans/out/extractor_return__20260221_165747.json`

## 1) Pipeline completo (treino + inferencia + contrato)
```mermaid
flowchart LR
  subgraph T["TREINO (bytes only)"]
    T1[PDFs de treino] --> T2[Janelas contiguas de 2 paginas]
    T2 --> T3[Features de bytes<br/>log_p1, log_p2, log_total,<br/>ratio_p2_p1, spread_max_min]
    T3 --> T4[KMeans fit]
    T4 --> T5[Centroides C1..Ck]
    T6[Tips textuais e anchors] --> T7[Score textual por cluster]
    T5 --> T7
    T7 --> T8[Cluster-alvo DESPACHO]
  end

  subgraph I["INFERENCIA (por PDF candidato)"]
    I1[PDF candidato] --> I2[Gera janelas 2p]
    I2 --> I3[Calcula features de bytes por janela]
    I3 --> I4[dist = distancia ao centroide-alvo]
    I4 --> I5[cluster_sim = 1/(1+dist)]
    I2 --> I6[size_total = media slot a slot<br/>min(model, cand)/max(...)]
    I5 --> I7[final = w_size*size_total + w_cluster*cluster_sim]
    I6 --> I7
    I7 --> I8[Escolhe maior final]
  end

  T8 --> I4
  I8 --> O1[detector_return]
  O1 --> O2[process_id]
  O1 --> O3[candidate_pdf]
  O1 --> O4[best_page_indices]
  O1 --> O5[final_score]
  O1 --> O6[bytes_ref usado]
  O1 --> O7[Extrator usa apenas best_page_indices]
```

## 2) Comparativo visual: robust x kmeans
```mermaid
flowchart TB
  subgraph R["ROBUST (baseline size-only)"]
    R1[Referencia por slot p1/p2] --> R2[Score por slot: min/max]
    R2 --> R3[Score janela = media(slot)]
    R3 --> R4[Thresholds min_p1, min_p2, min_any]
    R4 --> R5[Top-1 janela]
  end

  subgraph K["KMEANS (size + cluster_sim)"]
    K1[Features bytes por janela] --> K2[KMeans + centroide alvo]
    K2 --> K3[cluster_sim = 1/(1+dist)]
    K1 --> K4[size_total = media(slot min/max)]
    K3 --> K5[Score final ponderado]
    K4 --> K5
    K5 --> K6[Top-1 janela]
  end

  R5 --> C1[Comparar best_page_indices]
  K6 --> C1
  C1 --> C2[Comparar coverage no extrator]
```

## 3) Exemplo numerico simples (1 janela)
- Modelo esperado (slot): `p1=145000`, `p2=48000`
- Janela candidata: `p1=150000`, `p2=47000`
- Similaridade size por slot:
  - `s1 = 145000/150000 = 0.9667`
  - `s2 = 47000/48000 = 0.9792`
- `size_total = (s1+s2)/2 = 0.9730`
- Se `distancia ao centroide-alvo = 0.22`, entao:
  - `cluster_sim = 1/(1+0.22) = 0.8197`
- Com pesos `w_size=0.6`, `w_cluster=0.4`:
  - `final = 0.6*0.9730 + 0.4*0.8197 = 0.9117`

## 4) Resultado consolidado (Q completo)
- Detector:
  - `robust_detected = 210/242`
  - `kmeans_detected = 228/242`
  - `best_page_indices diferentes = 224/242`
- Extrator (coverage media por PDF):
  - `robust = 0.2317`
  - `kmeans = 0.2476` (delta `+0.0159`)
- FP proxy:
  - `robust = 0.7366`
  - `kmeans = 0.5956` (menor melhor)

## 5) Leitura operacional
- KMeans hoje treina em bytes, nao em texto.
- Texto entra apenas para apontar qual cluster representa despacho.
- O contrato com o extrator fica explicito no `detector_return` com as paginas escolhidas.
