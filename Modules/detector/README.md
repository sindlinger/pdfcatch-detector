# Detector Modules

Estrutura organizada por abordagem de deteccao.

Cada pasta abaixo representa um metodo do pipeline:

- `bytes/`: similaridade por tamanho de janela (size-only).
- `tlsh/`: similaridade por TLSH em bytes opacos da janela.
- `stream/`: similaridade por `/Contents` (simhash de stream).
- `text/`: similaridade textual (`text_hybrid`/`text_order`).
- `image/`: similaridade visual por `dHash`.

Implementacoes atuais (codigo-fonte em uso) continuam em:

- `src/pdfcatch/modules/pagebytes/prefilter.py`
- `src/pdfcatch/modules/stream/contents.py`
- `src/pdfcatch/modules/text/metrics.py`
- `src/pdfcatch/modules/image/dhash.py`
- `src/pdfcatch/modules/file/tlsh_prefilter.py` (quando presente no ambiente)

