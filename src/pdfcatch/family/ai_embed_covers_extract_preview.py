from pdfcatch.ia.ai_embed_covers_extract_preview import *  # noqa: F401,F403

if __name__ == "__main__":
    try:
        from pdfcatch.ia.ai_embed_covers_extract_preview import main as _main
    except Exception:
        raise SystemExit(1)
    raise SystemExit(_main())
