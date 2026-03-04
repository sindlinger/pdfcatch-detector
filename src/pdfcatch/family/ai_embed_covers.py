from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import numpy as np

from pdfcatch.family.features import extract_page_text_from_doc


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


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


def _read_cover_text(pdf: Path) -> str:
    try:
        doc = fitz.open(pdf)
    except Exception:
        return ""
    try:
        return _normalize_text(extract_page_text_from_doc(doc, 0))
    finally:
        try:
            doc.close()
        except Exception:
            pass


def run(
    *,
    dir_path: Path,
    out_path: Path,
    model_name: str,
    batch_size: int,
    min_chars: int,
    max_files: int,
) -> int:
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"diretório inválido: {dir_path}")

    pdfs = sorted(p for p in dir_path.glob("*.pdf") if p.is_file())
    if max_files > 0:
        pdfs = pdfs[: int(max_files)]
    if not pdfs:
        raise ValueError(f"nenhum PDF encontrado em: {dir_path}")

    files: list[str] = []
    texts: list[str] = []
    skipped: list[dict[str, Any]] = []
    for pdf in pdfs:
        txt = _read_cover_text(pdf)
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

    if not texts:
        raise ValueError(
            "nenhuma capa com texto suficiente para embeddings. "
            f"arquivos_lidos={len(pdfs)} min_chars={int(min_chars)}"
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=arr,
        files=np.array(files, dtype=object),
        model_name=np.array([str(model_name)], dtype=object),
        source_dir=np.array([str(dir_path.resolve())], dtype=object),
        created_at_utc=np.array([_now_utc_iso()], dtype=object),
    )

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta = {
        "schema": "ai_cover_embeddings.v1",
        "created_at_utc": _now_utc_iso(),
        "source_dir": str(dir_path.resolve()),
        "model_name": str(model_name),
        "device": "cuda",
        "cpu_fallback": False,
        "files_total": int(len(pdfs)),
        "files_embedded": int(len(files)),
        "files_skipped": int(len(skipped)),
        "embedding_dim": int(arr.shape[1]) if arr.ndim == 2 else None,
        "output_npz": str(out_path.resolve()),
        "skipped": skipped[:200],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "status": "ok",
        "output_npz": str(out_path.resolve()),
        "output_meta": str(meta_path.resolve()),
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
            "AI Cover Embeddings: extrai embeddings somente da página 0 "
            "de despachos validados (GPU obrigatória, sem classe negativa)"
        )
    )
    ap.add_argument(
        "--dir",
        required=True,
        help="Diretório que contém apenas PDFs de despachos validados",
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
    return run(
        dir_path=Path(args.dir),
        out_path=Path(args.out),
        model_name=str(args.model),
        batch_size=int(args.batch_size),
        min_chars=int(args.min_chars),
        max_files=int(args.max_files),
    )


if __name__ == "__main__":
    raise SystemExit(main())
