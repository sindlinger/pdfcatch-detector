# bytes

Abordagem: **size-only** (sem leitura semantica de conteudo).

Uso:

- compara o tamanho em bytes de janelas de paginas (`K` paginas) contra o modelo.
- gera score por relacao de tamanho (`min(a,b)/max(a,b)`).

Referencia atual do codigo:

- `src/pdfcatch/modules/pagebytes/prefilter.py`

