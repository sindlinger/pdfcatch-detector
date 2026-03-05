from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "API requer dependencias: fastapi e uvicorn. "
        "instale com: pip install -e ."
    ) from exc

from pdfcatch.catalogacao_classica import build_catalog as classica_build_catalog
from pdfcatch.catalogacao_classica import find_docs as classica_find_docs
from pdfcatch.catalogacao_classica._internal.catalog_store import (
    catalog_dir_for_tag as classica_catalog_dir_for_tag,
    normalize_tag as classica_normalize_tag,
)
from pdfcatch.qwen_supervisionado import build_catalog as qwen_build_catalog
from pdfcatch.qwen_supervisionado import find_docs as qwen_find_docs
from pdfcatch.qwen_supervisionado import fit_model as qwen_fit_model
from pdfcatch.qwen_supervisionado import predict_model as qwen_predict_model
from pdfcatch.qwen_supervisionado import prepare_dataset as qwen_prepare_dataset
from pdfcatch.qwen_supervisionado._internal.catalog_store import (
    catalog_dir_for_tag as qwen_catalog_dir_for_tag,
    normalize_tag as qwen_normalize_tag,
)
from pdfcatch.qwen_supervisionado._internal.paths import QWEN_MODEL_ROOT, QWEN_OUTPUT_ROOT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSICA_ROOT = PROJECT_ROOT / "io" / "catalogacao_classica"
CLASSICA_OUTPUT_ROOT = CLASSICA_ROOT / "outputs"


def _now_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _run_or_400(fn, **kwargs) -> None:
    try:
        rc = fn(**kwargs)
        if int(rc) != 0:
            raise HTTPException(status_code=500, detail=f"run retornou rc={rc}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class ClassicaBuildCatalogReq(BaseModel):
    tag: str
    dir: str
    update: bool = False
    model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    batch_size: int = 32
    min_chars: int = 40
    max_files: int = 0
    threshold_method: str = "p05"
    threshold_sigma: float = 2.0


class ClassicaFindDocsReq(BaseModel):
    tag: str
    dir: str
    page_index: int = 0
    min_chars: int = 40
    threshold_override_pct: float | None = None
    out: str | None = None


class QwenBuildCatalogReq(BaseModel):
    tag: str
    dir: str
    update: bool = False
    model: str
    batch_size: int = 32
    min_chars: int = 40
    max_files: int = 0
    threshold_method: str = "p05"
    threshold_sigma: float = 2.0


class QwenFindDocsReq(BaseModel):
    tag: str
    dir: str
    min_chars: int = 40
    threshold_override_pct: float | None = None
    out: str | None = None


class QwenPrepareDatasetReq(BaseModel):
    positive_dir: str
    negative_dir: str
    out_dataset_dir: str
    positive_label: str = "despacho"
    negative_label: str = "nao_despacho"
    seed: int = 42
    balance: bool = True
    train_pct: float = 0.8
    val_pct: float = 0.1
    test_pct: float = 0.1
    overwrite: bool = True


class QwenFitModelReq(BaseModel):
    dataset_dir: str
    embedding_model: str
    model_out_dir: str | None = None
    report_out: str | None = None
    min_chars: int = 20
    batch_size_embed: int = 8
    batch_size_train: int = 256
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    seed: int = 42
    max_text_chars: int = 3000
    max_seq_length: int = 384


class QwenPredictReq(BaseModel):
    model_dir: str
    pdf: str | None = None
    dir: str | None = None
    out: str | None = None
    batch_size_embed: int = 32


app = FastAPI(title="PDFCatch IA API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/catalogacao-classica/build-catalog")
def api_classica_build_catalog(req: ClassicaBuildCatalogReq) -> dict[str, Any]:
    _run_or_400(
        classica_build_catalog.run,
        tag=str(req.tag),
        dir_path=Path(req.dir),
        update=bool(req.update),
        model_name=str(req.model),
        batch_size=int(req.batch_size),
        min_chars=int(req.min_chars),
        max_files=int(req.max_files),
        threshold_method=str(req.threshold_method),
        threshold_sigma=float(req.threshold_sigma),
    )
    norm_tag = classica_normalize_tag(req.tag)
    cdir = classica_catalog_dir_for_tag(norm_tag)
    return {
        "status": "ok",
        "approach": "catalogacao_classica",
        "operation": "build_catalog",
        "tag": norm_tag,
        "catalog_dir": str(cdir.resolve()),
        "catalog_json": str((cdir / "catalog.json").resolve()),
    }


@app.post("/api/v1/catalogacao-classica/find-docs")
def api_classica_find_docs(req: ClassicaFindDocsReq) -> dict[str, Any]:
    out = (
        Path(req.out)
        if req.out
        else CLASSICA_OUTPUT_ROOT / f"find_docs_{classica_normalize_tag(req.tag)}_{_now_compact()}.json"
    )
    _ensure_parent(out)
    _run_or_400(
        classica_find_docs.run,
        tag=str(req.tag),
        dir_path=Path(req.dir),
        out_path=out,
        page_index=int(req.page_index),
        min_chars=int(req.min_chars),
        threshold_override_pct=req.threshold_override_pct,
    )
    payload = _read_json(out)
    return {
        "status": "ok",
        "approach": "catalogacao_classica",
        "operation": "find_docs",
        "out": str(out.resolve()),
        "total_files": payload.get("total_files"),
        "found_count": payload.get("found_count"),
    }


@app.post("/api/v1/qwen-supervisionado/build-catalog")
def api_qwen_build_catalog(req: QwenBuildCatalogReq) -> dict[str, Any]:
    _run_or_400(
        qwen_build_catalog.run,
        tag=str(req.tag),
        dir_path=Path(req.dir),
        update=bool(req.update),
        model_name=str(req.model),
        batch_size=int(req.batch_size),
        min_chars=int(req.min_chars),
        max_files=int(req.max_files),
        threshold_method=str(req.threshold_method),
        threshold_sigma=float(req.threshold_sigma),
    )
    norm_tag = qwen_normalize_tag(req.tag)
    cdir = qwen_catalog_dir_for_tag(norm_tag)
    return {
        "status": "ok",
        "approach": "qwen_supervisionado",
        "operation": "build_catalog",
        "tag": norm_tag,
        "catalog_dir": str(cdir.resolve()),
        "catalog_json": str((cdir / "catalog.json").resolve()),
    }


@app.post("/api/v1/qwen-supervisionado/find-docs")
def api_qwen_find_docs(req: QwenFindDocsReq) -> dict[str, Any]:
    out = (
        Path(req.out)
        if req.out
        else QWEN_OUTPUT_ROOT / f"find_docs_{qwen_normalize_tag(req.tag)}_{_now_compact()}.json"
    )
    _run_or_400(
        qwen_find_docs.run,
        tag=str(req.tag),
        dir_path=Path(req.dir),
        out_path=out,
        min_chars=int(req.min_chars),
        threshold_override_pct=req.threshold_override_pct,
    )
    payload = _read_json(out)
    return {
        "status": "ok",
        "approach": "qwen_supervisionado",
        "operation": "find_docs",
        "out": str(out.resolve()),
        "total_files": payload.get("total_files"),
        "found_count": payload.get("found_count"),
    }


@app.post("/api/v1/qwen-supervisionado/prepare-dataset")
def api_qwen_prepare_dataset(req: QwenPrepareDatasetReq) -> dict[str, Any]:
    _run_or_400(
        qwen_prepare_dataset.run,
        positive_dir=Path(req.positive_dir),
        negative_dir=Path(req.negative_dir),
        out_dataset_dir=Path(req.out_dataset_dir),
        positive_label=str(req.positive_label),
        negative_label=str(req.negative_label),
        seed=int(req.seed),
        balance=bool(req.balance),
        train_pct=float(req.train_pct),
        val_pct=float(req.val_pct),
        test_pct=float(req.test_pct),
        overwrite=bool(req.overwrite),
    )
    summary = _read_json(Path(req.out_dataset_dir) / "summary.json")
    return {
        "status": "ok",
        "approach": "qwen_supervisionado",
        "operation": "prepare_dataset",
        "dataset_dir": str(Path(req.out_dataset_dir).resolve()),
        "counts_selected": summary.get("counts_selected"),
        "counts_split": summary.get("counts_split"),
    }


@app.post("/api/v1/qwen-supervisionado/fit-model")
def api_qwen_fit_model(req: QwenFitModelReq) -> dict[str, Any]:
    ts = _now_compact()
    model_out_dir = Path(req.model_out_dir) if req.model_out_dir else QWEN_MODEL_ROOT / f"supervised_{Path(req.dataset_dir).name}_{ts}"
    report_out = Path(req.report_out) if req.report_out else QWEN_OUTPUT_ROOT / f"fit_model_{Path(req.dataset_dir).name}_{ts}.json"
    _run_or_400(
        qwen_fit_model.run,
        dataset_dir=Path(req.dataset_dir),
        model_out_dir=model_out_dir,
        report_out=report_out,
        embedding_model=str(req.embedding_model),
        min_chars=int(req.min_chars),
        batch_size_embed=int(req.batch_size_embed),
        batch_size_train=int(req.batch_size_train),
        epochs=int(req.epochs),
        learning_rate=float(req.learning_rate),
        weight_decay=float(req.weight_decay),
        dropout=float(req.dropout),
        seed=int(req.seed),
        max_text_chars=int(req.max_text_chars),
        max_seq_length=int(req.max_seq_length),
    )
    payload = _read_json(report_out)
    return {
        "status": "ok",
        "approach": "qwen_supervisionado",
        "operation": "fit_model",
        "model_out_dir": str(model_out_dir.resolve()),
        "report_out": str(report_out.resolve()),
        "best_epoch": payload.get("best_epoch"),
        "metrics_test": (payload.get("metrics") or {}).get("test"),
    }


@app.post("/api/v1/qwen-supervisionado/predict-model")
def api_qwen_predict_model(req: QwenPredictReq) -> dict[str, Any]:
    if bool(req.pdf) == bool(req.dir):
        raise HTTPException(status_code=400, detail="informe exatamente um: pdf ou dir")
    out = Path(req.out) if req.out else QWEN_OUTPUT_ROOT / f"predict_model_{_now_compact()}.json"
    _run_or_400(
        qwen_predict_model.run,
        model_dir=Path(req.model_dir),
        pdf=(Path(req.pdf) if req.pdf else None),
        dir_path=(Path(req.dir) if req.dir else None),
        out_path=out,
        batch_size_embed=int(req.batch_size_embed),
    )
    payload = _read_json(out)
    return {
        "status": "ok",
        "approach": "qwen_supervisionado",
        "operation": "predict_model",
        "out": str(out.resolve()),
        "total_files": payload.get("total_files"),
        "predicted_count": payload.get("predicted_count"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inicia API das duas abordagens de IA do pdfcatch.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args(argv)

    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "API requer uvicorn. instale com: pip install -e ."
        ) from exc
    uvicorn.run("pdfcatch.api_server:app", host=str(args.host), port=int(args.port), reload=bool(args.reload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

