"""
Pipeline de famílias (bytes-only) organizado em módulos:
- features: extração 11D por página
- catalog_registry: registro de TAG -> caminho do catálogo
- catalog_collect: gera CSV e resumo (política oficial: page_index=0)
- catalog_register: cria catálogo usando somente page_index=0 (family_catalog)
- catalog_build: implementação da catalogação (bloqueio de recatalogação + auditoria)
- similarity_search: usa catálogo já registrado para buscar semelhança e salvar resultados
- similarity_score: núcleo de pontuação estrutural + lexical (tokens + IA em GPU) por família
- ai_embed_covers: extrai embeddings de capa (página 0) para gabarito positivo IA
- ai_paradigm_validate: gera embedding paradigma e valida capa_vs_paradigma
- elbow: sugere k pelo método do cotovelo
"""
