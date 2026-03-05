from pdfcatch.catalogacao_classica._internal.compare_with_paradigm import *  # noqa: F401,F403

if __name__ == "__main__":
    try:
        from pdfcatch.catalogacao_classica._internal.compare_with_paradigm import main as _main
    except Exception:
        raise SystemExit(1)
    raise SystemExit(_main())
