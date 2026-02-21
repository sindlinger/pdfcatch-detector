# tlsh

Abordagem: **TLSH em bytes opacos de janela PDF**.

Uso:

- fingerprint TLSH do mini-PDF da janela (sem parse semantico).
- comparacao por distancia TLSH normalizada em score.

Referencia atual do codigo:

- `Modules/detector/bytes/window_fingerprint.py`
- pontos de orquestracao em `Modules/web/studio_app.py` e `src/pdfcatch/cli.py`
