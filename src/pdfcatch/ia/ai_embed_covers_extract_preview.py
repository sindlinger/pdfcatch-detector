from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import numpy as np

from pdfcatch.ia.features import extract_page_text_from_doc


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_local_human() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _now_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _log_step(step: int, total: int, message: str) -> None:
    print(f"[{_now_local_human()}] [{step}/{total}] {message}")


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _slug(name: str) -> str:
    out = []
    for ch in str(name or ""):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    return s or "arquivo"


def _load_sentence_transformer_gpu(model_name: str):
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise ValueError(
            "IA de embeddings exige GPU: pacote 'torch' não encontrado. "
            "instale torch com CUDA habilitado."
        ) from exc

    if not bool(torch.cuda.is_available()):
        raise ValueError(
            "IA de embeddings oficial exige GPU CUDA ativa. "
            "fallback para CPU não é permitido."
        )

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        raise ValueError(
            "IA de embeddings exige pacote 'sentence-transformers'."
        ) from exc

    try:
        model = SentenceTransformer(str(model_name), device="cuda")
    except Exception as exc:
        raise ValueError(
            f"falha ao carregar modelo de embeddings em GPU: model={model_name} erro={exc}"
        ) from exc
    return model


def _extract_selected_page_text_and_pdf(
    pdf: Path,
    page_index: int,
    extracted_out_path: Path,
) -> tuple[str, str | None]:
    try:
        doc = fitz.open(pdf)
    except Exception as exc:
        return "", f"open_error:{exc}"
    try:
        if page_index < 0 or page_index >= int(doc.page_count):
            return "", f"page_out_of_range(total_pages={int(doc.page_count)})"

        text = _normalize_text(extract_page_text_from_doc(doc, page_index))

        out_doc = fitz.open()
        try:
            out_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)
            extracted_out_path.parent.mkdir(parents=True, exist_ok=True)
            out_doc.save(extracted_out_path)
        finally:
            try:
                out_doc.close()
            except Exception:
                pass
        return text, None
    except Exception as exc:
        return "", f"extract_error:{exc}"
    finally:
        try:
            doc.close()
        except Exception:
            pass


def run(
    *,
    dir_path: Path,
    out_path: Path,
    cover_page: int,
    covers_out_dir: Path,
    only_extract: bool,
    model_name: str,
    batch_size: int,
    min_chars: int,
    max_files: int,
) -> int:
    steps_total = 6
    _log_step(1, steps_total, "Validando diretório e parâmetros")
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"diretório inválido: {dir_path}")
    if int(cover_page) < 0:
        raise ValueError(f"cover_page inválido (esperado >=0): {cover_page}")

    _log_step(2, steps_total, f"Listando PDFs do diretório: {dir_path.resolve()}")
    pdfs = sorted(p for p in dir_path.glob("*.pdf") if p.is_file())
    if max_files > 0:
        pdfs = pdfs[: int(max_files)]
    if not pdfs:
        raise ValueError(f"nenhum PDF encontrado em: {dir_path}")

    _log_step(
        3,
        steps_total,
        "Extraindo página informada como capa "
        f"(cover_page={int(cover_page)}, total_pdfs={len(pdfs)}) "
        f"para conferência em: {covers_out_dir.resolve()}",
    )
    files: list[str] = []
    texts: list[str] = []
    skipped: list[dict[str, Any]] = []
    extracted_files: list[str] = []
    total = len(pdfs)
    for idx, pdf in enumerate(pdfs, start=1):
        out_name = f"{idx:05d}__{_slug(pdf.stem)}__cover_p{int(cover_page)+1:04d}.pdf"
        extracted_path = covers_out_dir / out_name
        txt, err = _extract_selected_page_text_and_pdf(pdf, int(cover_page), extracted_path)
        if err is not None:
            _log_step(3, steps_total, f"extração {idx}/{total}: {pdf.name} -> ERRO ({err})")
            skipped.append({"file": pdf.name, "reason": err})
            continue

        _log_step(3, steps_total, f"extração {idx}/{total}: {pdf.name} -> {extracted_path.resolve()}")
        extracted_files.append(str(extracted_path.resolve()))

        if len(txt) < int(min_chars):
            skipped.append(
                {
                    "file": pdf.name,
                    "reason": f"cover_text_too_short({len(txt)}<{int(min_chars)})",
                }
            )
            continue
        files.append(pdf.name)
        texts.append(txt)

    if only_extract:
        _log_step(4, steps_total, "Modo only-extract ativo: encerrando após conferência humana")
        payload = {
            "status": "ok_extract_only",
            "cover_page": int(cover_page),
            "covers_out_dir": str(covers_out_dir.resolve()),
            "files_total": int(len(pdfs)),
            "files_extracted": int(len(extracted_files)),
            "files_skipped": int(len(skipped)),
            "steps_logged": True,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not texts:
        raise ValueError(
            "nenhuma capa com texto suficiente para embeddings. "
            f"arquivos_lidos={len(pdfs)} min_chars={int(min_chars)}"
        )

    _log_step(
        4,
        steps_total,
        "Gerando embeddings em GPU "
        f"(model={model_name}, batch_size={int(batch_size)}, files={len(texts)}, cover_page={int(cover_page)})",
    )
    model = _load_sentence_transformer_gpu(str(model_name))
    vectors = model.encode(
        texts,
        batch_size=max(1, int(batch_size)),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    arr = np.array(vectors, dtype=np.float32)

    _log_step(5, steps_total, f"Salvando embeddings em: {out_path.resolve()}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=arr,
        files=np.array(files, dtype=object),
        cover_page=np.array([int(cover_page)], dtype=np.int32),
        covers_out_dir=np.array([str(covers_out_dir.resolve())], dtype=object),
        model_name=np.array([str(model_name)], dtype=object),
        source_dir=np.array([str(dir_path.resolve())], dtype=object),
        created_at_utc=np.array([_now_utc_iso()], dtype=object),
    )

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta = {
        "schema": "ai_cover_embeddings.v1",
        "created_at_utc": _now_utc_iso(),
        "source_dir": str(dir_path.resolve()),
        "cover_page": int(cover_page),
        "covers_out_dir": str(covers_out_dir.resolve()),
        "model_name": str(model_name),
        "device": "cuda",
        "cpu_fallback": False,
        "files_total": int(len(pdfs)),
        "files_embedded": int(len(files)),
        "files_skipped": int(len(skipped)),
        "embedding_dim": int(arr.shape[1]) if arr.ndim == 2 else None,
        "output_npz": str(out_path.resolve()),
        "files_extracted": int(len(extracted_files)),
        "skipped": skipped[:200],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    _log_step(6, steps_total, f"Concluído. Diretório de capas para conferência: {covers_out_dir.resolve()}")
    payload = {
        "status": "ok",
        "cover_page": int(cover_page),
        "covers_out_dir": str(covers_out_dir.resolve()),
        "output_npz": str(out_path.resolve()),
        "output_meta": str(meta_path.resolve()),
        "steps_logged": True,
        "files_extracted": int(len(extracted_files)),
        "files_embedded": int(len(files)),
        "files_skipped": int(len(skipped)),
        "embedding_dim": meta.get("embedding_dim"),
        "model_name": str(model_name),
        "device": "cuda",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "AI Cover Embeddings + Preview: extrai página indicada como capa "
            "para conferência humana em tempo real e gera embeddings em GPU."
        )
    )
    ap.add_argument(
        "--dir",
        required=True,
        help="Diretório que contém apenas PDFs de despachos validados",
    )
    ap.add_argument(
        "--cover-page",
        type=int,
        required=True,
        help="Página que representa a capa (index zero-based)",
    )
    ap.add_argument(
        "--covers-out-dir",
        default=None,
        help="Diretório para salvar as capas extraídas para conferência humana",
    )
    ap.add_argument(
        "--only-extract",
        action="store_true",
        help="Extrai e salva capas para conferência, sem gerar embeddings",
    )
    ap.add_argument(
        "--out",
        default="io/out/despacho_cover_embeddings.npz",
        help="Arquivo .npz de saída",
    )
    ap.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="Modelo de embeddings (SentenceTransformer)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size para inferência na GPU",
    )
    ap.add_argument(
        "--min-chars",
        type=int,
        default=40,
        help="Mínimo de caracteres na capa para gerar embedding",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limite de arquivos (0 = sem limite)",
    )
    args = ap.parse_args(argv)
    covers_out_dir = (
        Path(args.covers_out_dir)
        if args.covers_out_dir
        else Path("io/out") / f"covers_preview_{_now_compact()}"
    )
    return run(
        dir_path=Path(args.dir),
        out_path=Path(args.out),
        cover_page=int(args.cover_page),
        covers_out_dir=covers_out_dir,
        only_extract=bool(args.only_extract),
        model_name=str(args.model),
        batch_size=int(args.batch_size),
        min_chars=int(args.min_chars),
        max_files=int(args.max_files),
    )


if __name__ == "__main__":
    raise SystemExit(main())
