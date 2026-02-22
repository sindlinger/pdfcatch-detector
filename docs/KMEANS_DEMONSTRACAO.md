# Demonstracao KMeans (Detector Bytes)

## Objetivo
Comparar `robust` vs `kmeans` no detector de despacho (2 paginas contiguas), usando o mesmo lote de entrada.

## Lote e arquivos usados
- Lote: `:Q` (242 PDFs)
- Detector robust return:
  - `/mnt/c/git/pdfcatch-detector/io/out_all_robust/detector_return__20260221_164107.json`
- Detector kmeans return:
  - `/mnt/c/git/pdfcatch-detector/io/out_all_kmeans/detector_return__20260221_165542.json`
- Extrator (current) em cima do robust:
  - `/mnt/c/git/pdfcatch-extractor/io_all_robust/out/extractor_return__20260221_165652.json`
- Extrator (current) em cima do kmeans:
  - `/mnt/c/git/pdfcatch-extractor/io_all_kmeans/out/extractor_return__20260221_165747.json`

## Como o KMeans funciona (hoje)
1. Gera janelas contiguas de 2 paginas para treino.
2. Calcula features de bytes por janela:
   - `log_p1`, `log_p2`, `log_total`, `ratio_p2_p1`, `spread_max_min`
3. Treina KMeans nessas features (texto nao entra no fit).
4. Usa texto apenas para escolher `cluster-alvo`:
   - termos obrigatorios + tips de ancoras.
5. Na inferencia, por janela candidata:
   - `size_total = media(min/max por pagina)`
   - `cluster_sim = 1/(1+distancia_ao_centroide_alvo)`
   - `final = w_size*size_total + w_cluster*cluster_sim`

## Como o Robust funciona (hoje)
1. Usa referencia de tamanho por slot (`p1`, `p2`) a partir dos templates.
2. Score por pagina: `si = min(model_i, cand_i)/max(...)`
3. Score da janela: media de `si`.
4. Aplica thresholds:
   - `p1 > min_p1`, `p2 >= min_p2`, `si >= min_any`.

## Comandos da rodada completa
```bash
cd /mnt/c/git/pdfcatch-detector
PYTHONPATH=src python -m pdfcatch.cli doc -d :Q -m bytes --bytes-ref-mode robust --top-n 1 --return --io-dir io/out_all_robust
PYTHONPATH=src python -m pdfcatch.cli doc -d :Q -m bytes --bytes-ref-mode kmeans --top-n 1 --return --io-dir io/out_all_kmeans

cd /mnt/c/git/pdfcatch-extractor
PYTHONPATH=src:. python -m pdfcatch_extractor.cli extract --from-return-path /mnt/c/git/pdfcatch-detector/io/out_all_robust/detector_return__20260221_164107.json --anchors configs/anchor_text_fields.json --match-mode current --io-dir io_all_robust
PYTHONPATH=src:. python -m pdfcatch_extractor.cli extract --from-return-path /mnt/c/git/pdfcatch-detector/io/out_all_kmeans/detector_return__20260221_165542.json --anchors configs/anchor_text_fields.json --match-mode current --io-dir io_all_kmeans
```

## Resultado objetivo (Q completo)
- Detector:
  - robust_detected = `210/242`
  - kmeans_detected = `228/242`
  - trocas de paginas (`best_page_indices`): `224/242` (alta discordancia)
- Extrator (coverage media por PDF):
  - robust = `0.2317`
  - kmeans = `0.2476` (`+0.0159`)
- FP proxy (heuristico):
  - robust = `0.7366`
  - kmeans = `0.5956` (melhor; menor risco proxy)

## Leitura pratica
- Se o objetivo principal for detectar mais despachos no lote misto, o `kmeans` ficou melhor.
- Se o objetivo for estabilidade de paginas entre metodos, a concordancia foi baixa.
- Para criterio final de FP real, precisa validacao manual (gabarito) em amostra.

## Artefatos de comparacao
- Resumo geral:
  - `/mnt/c/git/pdfcatch/io/out/compare_q_all_by_pdf_robust_vs_kmeans__20260221_1700.json`
- Tabela completa por arquivo:
  - `/mnt/c/git/pdfcatch/io/out/compare_q_all_by_pdf_rows__20260221_1702.csv`
- FP proxy:
  - `/mnt/c/git/pdfcatch/io/out/fp_proxy_q_all_robust_vs_kmeans__20260221_1704.json`
