"""Fraud-scoring web service.

Loads the model trained by train.py and serves it over HTTP with FastAPI.

    uvicorn predict:app --host 0.0.0.0 --port 9696

Endpoints
---------
GET  /health      -> service + model metadata
POST /predict     -> fraud probability and decision for one transaction
POST /score-file  -> score an uploaded CSV/Excel file

The scoring logic lives in `inference.py`, which has no web framework in it.
That lets the Streamlit UI reuse the exact same code when there is no backend
to call (see the cloud deployment section of the README).
"""
from __future__ import annotations

import io

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from inference import (MAX_UPLOAD_ROWS, FRAUD_TYPES, get_model, prepare,
                       score_frame, score_one)
from reasons import REQUIRED, detect_columns

MODEL = get_model()          # loads once at import time

app = FastAPI(title="PaySim Fraud Detection", version="1.0.0")


class Lead(BaseModel):  # transaction record
    step: int = Field(1, ge=1, le=744, description="Hour since the start of the simulation (1-744)")
    type: str = Field("TRANSFER", description="TRANSFER | CASH_OUT | PAYMENT | CASH_IN | DEBIT")
    amount: float = Field(0.0, ge=0.0)
    oldbalanceOrg: float = Field(0.0, ge=0.0, description="Sender balance BEFORE the transaction")
    oldbalanceDest: float = Field(0.0, ge=0.0, description="Recipient balance BEFORE the transaction")
    nameDest: str = Field("", description="Recipient id; ids starting with 'M' are merchants")


@app.get("/", include_in_schema=False)
def root():
    """Bare root URL used to return a bare 404 - send it to the docs instead."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": type(MODEL.model).__name__,
        "model_sha256": MODEL.sha256,
        "threshold": MODEL.threshold,
        "test_metrics": MODEL.metrics,
        "scored_types": sorted(FRAUD_TYPES),
    }


@app.post("/predict")
def predict(lead: Lead) -> dict:
    try:
        return score_one(MODEL, lead.model_dump())
    except Exception as exc:  # pragma: no cover - surface a clean 500
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.post("/score-file")
async def score_file(
    file: UploadFile = File(..., description="CSV or Excel of transactions"),
    limit: int = Query(50_000, ge=1, le=200_000,
                       description="max rows in the response"),
    include_legit: bool = Query(False, description="also return rows that were cleared"),
):
    """Score an uploaded CSV/Excel file and return the fraud cases found.

    The upload only needs the *pre-transaction* columns: step, type, amount,
    oldbalanceOrg, oldbalanceDest, nameDest. Post-transaction balances are
    ignored even if present, because they leak the label.
    """
    try:
        raw = await file.read()
        name = (file.filename or "").lower()
        buf = io.BytesIO(raw)
        try:
            df = pd.read_excel(buf) if name.endswith((".xlsx", ".xls")) else pd.read_csv(buf)
        except Exception as exc:
            # A file that isn't tabular blows up in the parser, well before we
            # ever look at columns - surface that as a 400, not a 500.
            raise HTTPException(
                400, f"could not parse '{file.filename}' as CSV or Excel — {exc}")

        if df.empty:
            raise HTTPException(400, "the uploaded file contains no rows")

        try:
            result = score_frame(df, MODEL, limit=limit,
                                 include_legit=include_legit,
                                 filename=file.filename)
        except ValueError as exc:
            # column-detection failure -> a helpful 400, not a 500
            raise HTTPException(400, str(exc))

        result["summary"]["engine"] = "fastapi"
        return result
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 9696)))
