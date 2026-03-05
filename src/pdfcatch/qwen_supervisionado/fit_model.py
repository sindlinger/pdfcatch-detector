from pdfcatch.qwen_supervisionado._internal.fit_model import *  # noqa: F401,F403

if __name__ == "__main__":
    try:
        from pdfcatch.qwen_supervisionado._internal.fit_model import main as _main
    except Exception:
        raise SystemExit(1)
    raise SystemExit(_main())
