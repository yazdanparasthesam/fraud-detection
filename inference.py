"""Model loading and scoring, with no web framework attached.

Shared by:

* `predict.py`       — the FastAPI service
* `streamlit_app.py` — the UI, when no API is reachable. Streamlit Community
  Cloud runs a single container with no backend beside it, so there `localhost:9696`
  does not exist. The UI falls back to scoring in-process here, which keeps one
  identical code path for both deployments.

Keeping this module free of FastAPI/pydantic means the UI can score without
pulling in a web framework it does not use.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from reasons import (REQUIRED, build_leaf_paths, detect_columns, humanise,
                     pick_trigger, readable_rule)

MODEL_PATH = Path(os.getenv("MODEL_PATH",
                            Path(__file__).with_name("models") / "fraud_model.bin"))

# Transaction types that can carry fraud. Anything else is auto-approved:
# in the PaySim data, PAYMENT / CASH_IN / DEBIT contain exactly zero fraud.
FRAUD_TYPES = {"TRANSFER", "CASH_OUT"}

MAX_UPLOAD_ROWS = 500_000      # guard rail: cap how much of one file we score

_BUNDLE = None


class ModelBundle:
    """The trained artefact plus everything derived from it once, at load."""

    def __init__(self, path=MODEL_PATH):
        path = Path(path)
        if not path.exists():
            raise RuntimeError(
                f"model not found at {path} - run `python train.py` first")
        with path.open("rb") as f:
            art = pickle.load(f)
        self.path = path
        self.artifact = art
        self.dv = art["dv"]
        self.model = art["model"]
        self.threshold = float(art.get("threshold", 0.5))
        self.metrics = art.get("metrics", {})
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.feature_names = list(self.dv.feature_names_)
        # Depth-4 tree => 13 leaves; resolve every decision path once.
        self.leaf_paths = build_leaf_paths(self.model, self.feature_names)


def get_model() -> ModelBundle:
    """Load once per process and reuse."""
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = ModelBundle()
    return _BUNDLE


# --------------------------------------------------------------------------- #
# Feature engineering - must mirror train.py `_features()` exactly
# --------------------------------------------------------------------------- #
def prepare(record: dict) -> dict:
    """Single-transaction mirror of train.py `_features()`.

    Only pre-transaction fields are used. The post-transaction balances
    (newbalanceOrig / newbalanceDest) are never requested: in PaySim a flagged
    transfer is cancelled, so those columns encode the outcome and leak the label.
    """
    step = int(record["step"])
    amount = float(record["amount"])
    obo = float(record["oldbalanceOrg"])
    obd = float(record["oldbalanceDest"])

    return {
        "step": step,
        "type": record["type"],
        "amount": amount,
        "oldbalanceOrg": obo,
        "oldbalanceDest": obd,
        "hour_of_day": step % 24,
        "day": step // 24,
        "log_amount": float(np.log1p(amount)),
        "log_oldbalance_org": float(np.log1p(obo)),
        "amount_to_balance": amount / (obo + 1.0),
        "balance_after_send": obo - amount,
        "orig_balance_zero": int(obo == 0.0),
        "dest_balance_zero": int(obd == 0.0),
        "dest_is_merchant": int(str(record["nameDest"]).startswith("M")),
        "amount_zero": int(amount == 0.0),
    }


def prepare_frame(df: pd.DataFrame, mapping: dict, warnings: list) -> pd.DataFrame:
    """Vectorised mirror of train.py `_features()` - only pre-transaction fields."""
    def col(canon, default):
        if canon in mapping:
            return pd.to_numeric(df[mapping[canon]], errors="coerce").fillna(default)
        warnings.append(f"no '{canon}' column - defaulting to {default}")
        return pd.Series(default, index=df.index, dtype="float64")

    step = col("step", 1.0).astype("int64")
    amount = col("amount", 0.0)
    obo = col("oldbalanceOrg", 0.0)
    obd = col("oldbalanceDest", 0.0)

    if "type" in mapping:
        txtype = df[mapping["type"]].astype(str).str.strip().str.upper()
    else:
        warnings.append("no 'type' column - treating every row as TRANSFER")
        txtype = pd.Series("TRANSFER", index=df.index)

    if "nameDest" in mapping:
        namedest = df[mapping["nameDest"]].astype(str)
    else:
        warnings.append("no 'nameDest' column - dest_is_merchant set to 0")
        namedest = pd.Series("", index=df.index)

    return pd.DataFrame({
        "step": step,
        "type": txtype,
        "amount": amount,
        "oldbalanceOrg": obo,
        "oldbalanceDest": obd,
        "hour_of_day": (step % 24).astype("int64"),
        "day": (step // 24).astype("int64"),
        "log_amount": np.log1p(amount),
        "log_oldbalance_org": np.log1p(obo),
        "amount_to_balance": amount / (obo + 1.0),
        "balance_after_send": obo - amount,
        "orig_balance_zero": (obo == 0.0).astype("int64"),
        "dest_balance_zero": (obd == 0.0).astype("int64"),
        "dest_is_merchant": namedest.str.startswith("M").astype("int64"),
        "amount_zero": (amount == 0.0).astype("int64"),
    })


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def risk_band(prob: float, threshold: float) -> str:
    if prob >= 0.9:
        return "critical"
    if prob >= threshold:
        return "high"
    if prob >= threshold / 10:
        return "medium"
    return "low"


def explain(bundle: ModelBundle, leaf_id: int) -> tuple[str, str]:
    """(rule, reason) for the leaf a transaction landed in."""
    conds = bundle.leaf_paths.get(int(leaf_id))
    if not conds:
        return "", ""
    return readable_rule(conds), humanise(*pick_trigger(conds))


def score_one(bundle: ModelBundle, record: dict) -> dict:
    """Score a single transaction dict. Same shape as `POST /predict`."""
    if record["type"] not in FRAUD_TYPES:
        # Structurally zero fraud in PAYMENT / CASH_IN / DEBIT.
        return {
            "fraud_probability": 0.0,
            "is_fraud": False,
            "decision": "auto_approve",
            "reason": f"type '{record['type']}' has no fraud in the training data",
            "threshold": bundle.threshold,
        }

    X = bundle.dv.transform([prepare(record)])
    probability = float(bundle.model.predict_proba(X)[0, 1])
    is_fraud = probability >= bundle.threshold
    return {
        "fraud_probability": round(probability, 6),
        "is_fraud": bool(is_fraud),
        "decision": "block" if is_fraud else "allow",
        "risk_band": risk_band(probability, bundle.threshold),
        "threshold": bundle.threshold,
    }


def score_frame(df: pd.DataFrame, bundle: ModelBundle | None = None,
                limit: int = 50_000, include_legit: bool = False,
                filename: str = "upload") -> dict:
    """Score a whole dataframe of raw transactions.

    Returns the same {"summary": ..., "rows": [...]} shape as `POST /score-file`,
    so the UI does not care whether it came from the API or from here.
    """
    bundle = bundle or get_model()
    warnings: list = []
    truncated = False
    if len(df) > MAX_UPLOAD_ROWS:
        df = df.head(MAX_UPLOAD_ROWS)
        truncated = True
        warnings.append(f"only the first {MAX_UPLOAD_ROWS:,} rows were scored")

    mapping = detect_columns(df)
    missing = [c for c in REQUIRED if c not in mapping]
    if missing:
        raise ValueError(
            f"could not find required column(s) {missing}. "
            f"Columns seen in the file: {list(df.columns)[:25]}")

    feats = prepare_frame(df, mapping, warnings)

    # Only TRANSFER / CASH_OUT can carry fraud; the rest are auto-approved.
    scoreable = feats["type"].isin(FRAUD_TYPES).values
    prob = np.zeros(len(df), dtype="float64")
    leaf = np.full(len(df), -1, dtype="int64")

    if scoreable.any():
        X = bundle.dv.transform(feats.loc[scoreable].to_dict(orient="records"))
        prob[scoreable] = bundle.model.predict_proba(X)[:, 1]
        leaf[scoreable] = bundle.model.apply(X)

    is_fraud = prob >= bundle.threshold
    band = np.where(prob >= 0.9, "critical",
                    np.where(is_fraud, "high",
                             np.where(prob >= bundle.threshold / 10, "medium", "low")))
    decision = np.where(is_fraud, "block",
                        np.where(scoreable, "allow", "auto_approve"))

    rules, why = [], []
    for i in range(len(df)):
        if not scoreable[i]:
            rules.append("")
            why.append(f"type '{feats['type'].iloc[i]}' carries no fraud in the training data")
        else:
            r, w = explain(bundle, leaf[i])
            rules.append(r)
            why.append(w)

    out = pd.DataFrame({
        "row": np.arange(1, len(df) + 1),
        "type": feats["type"].values,
        "amount": feats["amount"].values.round(2),
        "oldbalanceOrg": feats["oldbalanceOrg"].values.round(2),
        "oldbalanceDest": feats["oldbalanceDest"].values.round(2),
        "balance_after_send": feats["balance_after_send"].values.round(2),
        "fraud_probability": prob.round(6),
        "is_fraud": is_fraud,
        "decision": decision,
        "risk_band": band,
        "decision_rule": rules,
        "why_flagged": why,
    })
    if "nameDest" in mapping:
        out.insert(5, "nameDest", df[mapping["nameDest"]].values)
    if "isFraud" in mapping:
        out["actual_is_fraud"] = pd.to_numeric(
            df[mapping["isFraud"]], errors="coerce").fillna(0).astype(int).values

    flagged = out[out["is_fraud"]].sort_values("fraud_probability", ascending=False)
    returned = out if include_legit else flagged
    if include_legit:
        returned = returned.sort_values("fraud_probability", ascending=False)
    returned = returned.head(limit)

    summary = {
        "filename": filename,
        "total_rows": int(len(df)),
        "scored_rows": int(scoreable.sum()),
        "auto_approved_rows": int((~scoreable).sum()),
        "flagged_fraud": int(is_fraud.sum()),
        "flagged_rate_pct": round(100 * float(is_fraud.mean()), 4),
        "returned_rows": int(len(returned)),
        "truncated": truncated,
        "columns_detected": mapping,
        "warnings": warnings,
        "threshold": bundle.threshold,
        # lets the UI show which engine produced the result
        "engine": "in-process",
    }

    if "isFraud" in mapping:
        from sklearn.metrics import average_precision_score, roc_auc_score
        y = out["actual_is_fraud"].values
        m = scoreable
        if m.sum() and y[m].sum():
            tp = int((is_fraud[m] & (y[m] == 1)).sum())
            fp = int((is_fraud[m] & (y[m] == 0)).sum())
            fn = int((~is_fraud[m] & (y[m] == 1)).sum())
            summary["evaluation"] = {
                "actual_fraud": int(y.sum()),
                "true_positives": tp, "false_positives": fp, "false_negatives": fn,
                "precision": round(tp / max(tp + fp, 1), 4),
                "recall": round(tp / max(tp + fn, 1), 4),
                "pr_auc": round(float(average_precision_score(y[m], prob[m])), 4),
                "roc_auc": round(float(roc_auc_score(y[m], prob[m])), 4),
            }

    return {"summary": summary, "rows": returned.to_dict(orient="records")}
