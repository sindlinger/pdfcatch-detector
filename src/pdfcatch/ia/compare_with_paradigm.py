from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import fitz
import numpy as np

from pdfcatch.ia.features import extract_page_text_from_doc

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _load_sentence_transformer_gpu(model_name: str):
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise ValueError(
            "comparação com paradigma exige GPU: pacote 'torch' não encontrado. "
            "instale torch com CUDA habilitado."
        ) from exc

    if not bool(torch.cuda.is_available()):
        raise ValueError(
            "comparação com paradigma oficial exige GPU CUDA ativa. "
            "fallback para CPU não é permitido."
        )

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        raise ValueError(
            "comparação com paradigma exige pacote 'sentence-transformers'."
        ) from exc

    try:
        return SentenceTransformer(str(model_name), device="cuda")
    except Exception as exc:
        raise ValueError(
            f"falha ao carregar modelo em GPU: model={model_name} erro={exc}"
        ) from exc


def _load_paradigm(npz_path: Path) -> tuple[np.ndarray, float | None, str | None]:
    if not npz_path.exists() or not npz_path.is_file():
        raise ValueError(f"arquivo de paradigma não encontrado: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        if "paradigm_embedding" not in data:
            raise ValueError(f"npz inválido: campo 'paradigm_embedding' ausente em {npz_path}")
        paradigm = np.asarray(data["paradigm_embedding"], dtype=np.float32)
        if paradigm.ndim != 1:
            raise ValueError(f"paradigma inválido: esperado vetor 1D, recebido shape={tuple(paradigm.shape)}")

        threshold: float | None = None
        if "threshold" in data and len(data["threshold"]) > 0:
            try:
                threshold = float(np.asarray(data["threshold"]).reshape(-1)[0])
            except Exception:
                threshold = None

        source_embeddings: str | None = None
        if "source_embeddings" in data and len(data["source_embeddings"]) > 0:
            source_embeddings = str(data["source_embeddings"][0])

    norm = float(np.linalg.norm(paradigm))
    if norm <= 0.0:
        raise ValueError("paradigma inválido: norma zero")

    return (paradigm / norm).astype(np.float32), threshold, source_embeddings


def _infer_model_name(model_name: str | None, source_embeddings: str | None) -> str:
    if model_name and str(model_name).strip():
        return str(model_name).strip()

    if source_embeddings:
        p = Path(source_embeddings)
        if p.exists() and p.is_file():
            try:
                with np.load(p, allow_pickle=True) as data:
                    if "model_name" in data and len(data["model_name"]) > 0:
                        s = str(data["model_name"][0]).strip()
                        if s:
                            return s
            except Exception:
                pass

    return DEFAULT_MODEL


def _read_page_text(pdf_path: Path, page_index: int) -> str:
    if not pdf_path.exists() or not pdf_path.is_file():
        raise ValueError(f"pdf não encontrado: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise ValueError(f"falha ao abrir PDF: {pdf_path} erro={exc}") from exc

    try:
        if page_index < 0 or page_index >= int(doc.page_count):
            raise ValueError(
                f"page_index fora do intervalo: page_index={page_index} total_pages={int(doc.page_count)}"
            )
        return _normalize_text(extract_page_text_from_doc(doc, page_index))
    finally:
        try:
            doc.close()
        except Exception:
            pass


def run(
    *,
    pdf_path: Path,
    paradigm_npz: Path,
    page_index: int,
    model_name: str | None,
    min_chars: int,
) -> int:
    paradigm, threshold, source_embeddings = _load_paradigm(paradigm_npz)
    resolved_model = _infer_model_name(model_name, source_embeddings)

    page_text = _read_page_text(pdf_path, int(page_index))
    if len(page_text) < int(min_chars):
        payload = {
            "status": "insufficient_text",
            "pdf": str(pdf_path.resolve()),
            "page_index": int(page_index),
            "chars": int(len(page_text)),
            "min_chars": int(min_chars),
            "paradigm_npz": str(paradigm_npz.resolve()),
            "model_name": resolved_model,
            "threshold_pct": (round(float(threshold) * 100.0, 4) if threshold is not None else None),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    model = _load_sentence_transformer_gpu(resolved_model)
    vec = model.encode(
        [page_text],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    sim = float(np.dot(v, paradigm))

    payload = {
        "status": "ok",
        "pdf": str(pdf_path.resolve()),
        "page_index": int(page_index),
        "chars": int(len(page_text)),
        "model_name": resolved_model,
        "paradigm_npz": str(paradigm_npz.resolve()),
        "source_embeddings": source_embeddings,
        "similarity": sim,
        "similarity_pct": round(sim * 100.0, 4),
        "threshold": threshold,
        "threshold_pct": (round(float(threshold) * 100.0, 4) if threshold is not None else None),
        "pass_threshold": (bool(sim >= float(threshold)) if threshold is not None else None),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Compara uma única página de um PDF com o paradigma de despacho já gerado "
            "(sem pipeline de inferência)."
        )
    )
    ap.add_argument("--pdf", required=True, help="PDF a comparar")
    ap.add_argument(
        "--paradigm",
        default="io/out/despacho_paradigm_embedding.npz",
        help="Arquivo de paradigma .npz (gerado no ai_paradigm_validate)",
    )
    ap.add_argument(
        "--page-index",
        type=int,
        default=0,
        help="Página para comparar (0-based). Oficial: 0",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Modelo de embedding (opcional; se vazio tenta inferir do npz de origem)",
    )
    ap.add_argument(
        "--min-chars",
        type=int,
        default=40,
        help="Mínimo de caracteres para comparar",
    )
    args = ap.parse_args(argv)
    return run(
        pdf_path=Path(args.pdf),
        paradigm_npz=Path(args.paradigm),
        page_index=int(args.page_index),
        model_name=(str(args.model) if args.model else None),
        min_chars=int(args.min_chars),
    )


if __name__ == "__main__":
    raise SystemExit(main())
