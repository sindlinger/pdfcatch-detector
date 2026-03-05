from pdfcatch.ia.similarity_score import *  # noqa: F401,F403

if __name__ == "__main__":
    try:
        from pdfcatch.ia.similarity_score import main as _main
    except Exception:
        raise SystemExit(1)
    raise SystemExit(_main())
