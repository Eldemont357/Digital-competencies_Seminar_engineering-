from __future__ import annotations

import time
from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from .core import compute_quality_flags, missing_table, summarize_dataset

app = FastAPI(title="eda-cli API", version="0.1.0")


# =========================
# Common OpenAPI docs for CSV errors (so "400 is visible")
# =========================
CSV_400_RESPONSES = {
    400: {
        "description": "Bad Request: empty CSV or failed to parse",
        "content": {"application/json": {"example": {"detail": "Empty CSV file"}}},
    }
}


# =========================
# In-memory metrics (optional but nice)
# =========================
_METRICS = {
    "total_requests": 0,
    "total_latency_ms": 0.0,
    "last_ok_for_model": None,  # bool | None
}


def _metrics_add(latency_ms: float, ok_for_model: Optional[bool] = None) -> None:
    _METRICS["total_requests"] += 1
    _METRICS["total_latency_ms"] += float(latency_ms)
    if ok_for_model is not None:
        _METRICS["last_ok_for_model"] = bool(ok_for_model)


# =========================
# Helpers
# =========================
def _read_csv_upload(file: UploadFile) -> pd.DataFrame:
    try:
        content = file.file.read()

        # empty file OR file with only whitespace/newlines
        if not content or not content.strip():
            raise HTTPException(status_code=400, detail="Empty CSV file")

        df = pd.read_csv(BytesIO(content))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # after parsing must have both rows and cols
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="CSV has no rows or no columns")

    return df


def _bool_flags(flags_all: Dict[str, Any]) -> Dict[str, bool]:
    """Keep only boolean flags."""
    return {k: bool(v) for k, v in flags_all.items() if isinstance(v, bool)}


def _missing_to_json(miss: Any) -> Any:
    # missing_table often returns a DataFrame; convert to JSON-friendly
    try:
        return miss.to_dict(orient="records")  # DataFrame
    except Exception:
        try:
            return miss.to_dict()  # Series/dict-like
        except Exception:
            return str(miss)


# =========================
# Models
# =========================
class QualityRequest(BaseModel):
    n_rows: int = Field(..., ge=0)
    n_cols: int = Field(..., ge=0)
    missing_share: float = Field(..., ge=0.0, le=1.0)


class QualityResponse(BaseModel):
    ok_for_model: bool
    quality_score: float
    latency_ms: float
    flags: Dict[str, Any]


class QualityFromCsvResponse(BaseModel):
    ok_for_model: bool
    quality_score: float
    latency_ms: float
    flags: Dict[str, bool]
    n_rows: int
    n_cols: int


# =========================
# Endpoints
# =========================
@app.get("/health", tags=["system"])
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics", tags=["system"])
def metrics() -> Dict[str, Any]:
    total = int(_METRICS["total_requests"])
    avg = (_METRICS["total_latency_ms"] / total) if total > 0 else 0.0
    return {
        "total_requests": total,
        "avg_latency_ms": float(avg),
        "last_ok_for_model": _METRICS["last_ok_for_model"],
    }


# ---- Required: /quality ----
@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()

    flags = {
        "too_few_rows": req.n_rows < 50,
        "too_many_missing": req.missing_share > 0.3,
    }

    quality_score = 1.0
    if flags["too_few_rows"]:
        quality_score -= 0.4
    if flags["too_many_missing"]:
        quality_score -= 0.6

    quality_score = max(0.0, min(1.0, quality_score))
    ok_for_model = quality_score >= 0.7

    latency_ms = (time.perf_counter() - t0) * 1000.0
    _metrics_add(latency_ms, ok_for_model)

    return {
        "ok_for_model": ok_for_model,
        "quality_score": float(quality_score),
        "latency_ms": float(latency_ms),
        "flags": flags,
    }


# ---- Required: /quality-from-csv ----
@app.post(
    "/quality-from-csv",
    response_model=QualityFromCsvResponse,
    responses=CSV_400_RESPONSES,
    tags=["quality"],
)
def quality_from_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    t0 = time.perf_counter()

    df = _read_csv_upload(file)

    summary = summarize_dataset(df)
    miss = missing_table(df)
    flags_all = compute_quality_flags(summary, miss)

    score_raw = flags_all.get("quality_score", 0.0)
    try:
        quality_score = float(score_raw)
    except Exception:
        quality_score = 0.0

    quality_score = max(0.0, min(1.0, quality_score))
    ok_for_model = quality_score >= 0.7

    latency_ms = (time.perf_counter() - t0) * 1000.0
    _metrics_add(latency_ms, ok_for_model)

    return {
        "ok_for_model": ok_for_model,
        "quality_score": float(quality_score),
        "latency_ms": float(latency_ms),
        "flags": _bool_flags(flags_all),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
    }


# ---- HW04 custom endpoint: flags only ----
@app.post(
    "/quality-flags-from-csv",
    responses=CSV_400_RESPONSES,
    tags=["hw04-custom"],
)
def quality_flags_from_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    df = _read_csv_upload(file)

    summary = summarize_dataset(df)
    miss = missing_table(df)
    flags_all = compute_quality_flags(summary, miss)

    return {"flags": _bool_flags(flags_all)}


# ---- HW04 custom endpoint: full JSON summary ----
@app.post(
    "/summary-from-csv",
    responses=CSV_400_RESPONSES,
    tags=["hw04-custom"],
)
def summary_from_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    df = _read_csv_upload(file)

    summary = summarize_dataset(df)
    miss = missing_table(df)
    flags_all = compute_quality_flags(summary, miss)

    return {
        "shape": {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])},
        "summary": summary,
        "missing": _missing_to_json(miss),
        "flags": flags_all,  # здесь можно оставить полный набор (не только bool)
    }


# ---- Nice utilities over CSV ----
@app.post(
    "/missing-table-from-csv",
    responses=CSV_400_RESPONSES,
    tags=["utils"],
)
def missing_table_from_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    df = _read_csv_upload(file)
    miss = missing_table(df)
    return {"missing": _missing_to_json(miss)}


@app.post(
    "/head-from-csv",
    responses=CSV_400_RESPONSES,
    tags=["utils"],
)
def head_from_csv(
    file: UploadFile = File(...),
    n: int = Query(5, ge=1, le=100),
) -> Dict[str, Any]:
    df = _read_csv_upload(file)
    return {"head": df.head(int(n)).to_dict(orient="records")}


@app.post(
    "/sample-from-csv",
    responses=CSV_400_RESPONSES,
    tags=["utils"],
)
def sample_from_csv(
    file: UploadFile = File(...),
    n: int = Query(5, ge=1, le=100),
    random_state: int = Query(42, ge=0, le=10_000_000),
) -> Dict[str, Any]:
    df = _read_csv_upload(file)
    n = min(int(n), int(df.shape[0]))
    return {"sample": df.sample(n=n, random_state=int(random_state)).to_dict(orient="records")}
