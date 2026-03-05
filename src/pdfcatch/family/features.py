from pdfcatch.ia.features import *  # noqa: F401,F403

if __name__ == "__main__":
    try:
        from pdfcatch.ia.features import main as _main
    except Exception:
        raise SystemExit(1)
    raise SystemExit(_main())
