from __future__ import annotations

import time
from io import BytesIO
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from .core import compute_quality_flags, missing_table, summarize_dataset

app = FastAPI(title="eda-cli API", version="0.1.0")


def _read_csv_upload(file: UploadFile) -> pd.DataFrame:
    try:
        content = file.file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty CSV file")

        df = pd.read_csv(BytesIO(content))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # если после парсинга вообще ничего нет
    if df.shape[0] == 0 and df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Empty CSV after parsing")

    # чаще всего достаточно df.empty, но оставим и эту ветку
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file has no rows")

    return df


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _bool_flags(flags_all: dict[str, Any]) -> dict[str, bool]:
    """Оставляем только булевы флаги, как в эталонном api.py."""
    return {k: bool(v) for k, v in flags_all.items() if isinstance(v, bool)}


@app.post("/quality-from-csv")
def quality_from_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    t0 = time.perf_counter()

    df = _read_csv_upload(file)
    summary = summarize_dataset(df)
    miss = missing_table(df)
    flags_all = compute_quality_flags(summary, miss)

    # Берём скор из ядра, если он там есть (как в учебном репо)
    score_raw = flags_all.get("quality_score", 0.0)
    try:
        quality_score = float(score_raw)
    except Exception:
        quality_score = 0.0

    quality_score = max(0.0, min(1.0, quality_score))
    ok_for_model = quality_score >= 0.7  # как в эталоне

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "ok_for_model": ok_for_model,
        "quality_score": quality_score,
        "latency_ms": float(latency_ms),
        "flags": _bool_flags(flags_all),  # только bool
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
    }


# === HW04 REQUIRED ENDPOINT ===
@app.post("/quality-flags-from-csv")
def quality_flags_from_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    df = _read_csv_upload(file)

    summary = summarize_dataset(df)
    miss = missing_table(df)
    flags_all = compute_quality_flags(summary, miss)

    # строго только булевы флаги 
    return {"flags": _bool_flags(flags_all)}
