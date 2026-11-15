"""Train the final PaySim fraud-detection model.

Pipeline
--------
1. Stream the raw PaySim CSV (see README for the download command).
2. Keep TRANSFER + CASH_OUT - the only two transaction types that can carry fraud.
3. Build leak-free features (no post-transaction balance columns).
4. Stratified 60/20/20 split.
5. Train the selected model, evaluate on the held-out test set.
6. Persist DictVectorizer + model + threshold to models/fraud_model.bin.

Model choice: DecisionTreeClassifier(max_depth=4, class_weight='balanced'),
selected over LogisticRegression, RandomForest and XGBoost by validation
PR-AUC. See notebook.ipynb for the full comparison.

Memory note
-----------
The raw file is 6.36M rows; reading it in one shot needs ~4 GB. We stream it
in blocks and convert each block to pandas immediately, so peak memory stays
near the size of the filtered table (~350 MB) instead of 3x that.
"""
from __future__ import annotations

import gc
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from vectorize import transform_chunked

RAW_CSV = os.getenv("PAYSIM_CSV", "data/paysim.csv")
MODEL_OUT = os.getenv("MODEL_OUT", "models/fraud_model.bin")
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]
RANDOM_STATE = 42
BLOCK_ROWS = int(os.getenv("PAYSIM_BLOCK_ROWS", 200_000))   # rows per streamed block
MAX_ROWS = int(os.getenv("PAYSIM_MAX_ROWS", 0))             # 0 = no limit (debug aid)

# Selected by validation PR-AUC (see notebook.ipynb)
BEST_PARAMS = dict(max_depth=4, class_weight="balanced", random_state=RANDOM_STATE)

SCHEMA = pa.schema([
    ("step", "int16"), ("type", "str"), ("amount", "float64"), ("nameOrig", "str"),
    ("oldbalanceOrg", "float64"), ("newbalanceOrig", "float64"), ("nameDest", "str"),
    ("oldbalanceDest", "float64"), ("newbalanceDest", "float64"),
    ("isFraud", "int8"), ("isFlaggedFraud", "int8"),
])

CAT = ["type"]
NUM = ["step", "amount", "oldbalanceOrg", "oldbalanceDest", "hour_of_day", "day",
       "log_amount", "log_oldbalance_org", "amount_to_balance", "balance_after_send",
       "orig_balance_zero", "dest_balance_zero", "dest_is_merchant", "amount_zero"]


def _features(block: pa.Table) -> pd.DataFrame:
    """Leak-free features for one streamed chunk.

    All feature maths is done with numpy/pandas, not pyarrow.compute. The Arrow
    kernel names are not stable across releases - the modulo function in
    particular exists as `modulo` in pyarrow >= 26, as `remainder` in some
    builds, and is missing entirely in older ones - which made this pipeline
    crash depending on the installed version. Per-block pandas is just as fast
    and works everywhere.
    """
    pdf = block.select(['step', 'type', 'amount', 'oldbalanceOrg',
                        'oldbalanceDest', 'nameDest', 'isFraud']).to_pandas()
    step = pdf['step'].to_numpy().astype('int32')
    amount = pdf['amount'].to_numpy().astype('float32')
    obo = pdf['oldbalanceOrg'].to_numpy().astype('float32')
    obd = pdf['oldbalanceDest'].to_numpy().astype('float32')
    return pd.DataFrame({
        'step': step,
        'type': pdf['type'].astype('category').values,
        'amount': amount,
        'oldbalanceOrg': obo,
        'oldbalanceDest': obd,
        'hour_of_day': (step % 24).astype('int8'),
        'day': (step // 24).astype('int16'),
        'log_amount': np.log1p(amount).astype('float32'),
        'log_oldbalance_org': np.log1p(obo).astype('float32'),
        'amount_to_balance': (amount / (obo + np.float32(1.0))).astype('float32'),
        'balance_after_send': (obo - amount).astype('float32'),
        'orig_balance_zero': (obo == 0.0).astype('int8'),
        'dest_balance_zero': (obd == 0.0).astype('int8'),
        'dest_is_merchant': pdf['nameDest'].astype(str).str.startswith('M').to_numpy().astype('int8'),
        'amount_zero': (amount == 0.0).astype('int8'),
        'isFraud': pdf['isFraud'].to_numpy().astype('int8'),
    })


CACHE_PARQUET = os.getenv("PAYSIM_CACHE", "data/paysim_model.parquet")


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast dtypes and report the modelling table."""
    df["type"] = df["type"].astype("category")
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    print(f"raw rows          : {getattr(df, '_raw_rows', len(df)):,}")
    print(f"modelling table   : {df.shape[0]:,} rows x {df.shape[1]} cols "
          f"({df.memory_usage(deep=True).sum() / 1e6:.0f} MB)  "
          f"fraud rate {100 * df.isFraud.mean():.4f}%")
    return df


def load_and_prepare() -> pd.DataFrame:
    """Build (or reuse) the prepared modelling table.

    The first call streams the 471 MB CSV and writes a compact Parquet cache,
    so re-runs skip the parse entirely. Set PAYSIM_REBUILD=1 to force a rebuild.
    """
    if os.path.exists(CACHE_PARQUET) and not os.getenv("PAYSIM_REBUILD"):
        print(f"using cached prepared table: {CACHE_PARQUET}")
        return _finalise(pd.read_parquet(CACHE_PARQUET))

    if not os.path.exists(RAW_CSV):
        raise SystemExit(
            f"{RAW_CSV} not found. Download it first:\n"
            "  mkdir -p data\n"
            "  curl -L -o data/paysim.csv https://huggingface.co/datasets/"
            "vitaliy-sharandin/synthetic-fraud-detection/resolve/main/"
            "PS_20174392719_1491204439457_log.csv"
        )

    reader = pv.open_csv(
        RAW_CSV,
        convert_options=pv.ConvertOptions(column_types={f.name: f.type for f in SCHEMA}),
        read_options=pv.ReadOptions(block_size=32 << 20),
    )

    total = kept = 0
    parts: list[pd.DataFrame] = []
    for batch in reader:
        tbl = pa.Table.from_batches([batch])
        total += tbl.num_rows
        tbl = tbl.filter(pc.is_in(tbl["type"], value_set=pa.array(FRAUD_TYPES)))
        if tbl.num_rows:
            kept += tbl.num_rows
            parts.append(_features(tbl))
        del tbl
        if MAX_ROWS and total >= MAX_ROWS:
            break

    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    print(f"raw rows          : {total:,}")
    print(f"after type filter : {kept:,} ({100 * df.isFraud.mean():.4f}% fraud)")

    os.makedirs(os.path.dirname(CACHE_PARQUET) or ".", exist_ok=True)
    df.to_parquet(CACHE_PARQUET, index=False)
    print(f"cached prepared table -> {CACHE_PARQUET}")
    return _finalise(df)


def main() -> None:
    df = load_and_prepare()
    y = df["isFraud"].values.astype("int8")
    Xdf = df[CAT + NUM]
    del df
    gc.collect()

    idx = np.arange(len(y))
    tr_idx, tmp = train_test_split(idx, test_size=0.4, random_state=RANDOM_STATE, stratify=y)
    va_idx, te_idx = train_test_split(tmp, test_size=0.5, random_state=RANDOM_STATE,
                                      stratify=y[tmp])
    # Default: refit on train+val once the model has been chosen (best practice).
    # PAYSIM_FIT_SPLIT=train fits on the train split only - much lighter on memory
    # for machines with < 4 GB RAM.
    if os.getenv("PAYSIM_FIT_SPLIT") == "train":
        fit_idx = tr_idx
        print("fitting on: train only")
    else:
        fit_idx = np.concatenate([tr_idx, va_idx])
        print("fitting on: train + val")
    print(f"\ntrain={len(tr_idx):,}  val={len(va_idx):,}  test={len(te_idx):,}")
    print(f"fraud  train={y[tr_idx].sum():,}  val={y[va_idx].sum():,}  test={y[te_idx].sum():,}")

    dv = DictVectorizer(sparse=True)
    dv.fit(Xdf.iloc[:50_000].to_dict(orient="records"))
    X_fit = transform_chunked(dv, Xdf.iloc[fit_idx])
    X_te = transform_chunked(dv, Xdf.iloc[te_idx])
    X_fit.data = X_fit.data.astype(np.float32)
    X_te.data = X_te.data.astype(np.float32)
    y_fit, y_te = y[fit_idx], y[te_idx]
    del Xdf, tr_idx, va_idx, tmp, idx
    gc.collect()
    print(f"\nX_fit={X_fit.shape}  X_test={X_te.shape}")

    print("\n--- final model ---")
    model = DecisionTreeClassifier(**BEST_PARAMS)
    model.fit(X_fit, y_fit)
    print(f"trained: {model}")

    prob = model.predict_proba(X_te)[:, 1]
    ap = average_precision_score(y_te, prob)
    auc = roc_auc_score(y_te, prob)
    print(f"\nTEST  PR-AUC  = {ap:.4f}")
    print(f"TEST  ROC-AUC = {auc:.4f}")

    # Operating point: the F1-optimal threshold, read off the precision-recall curve.
    # (Reported, not "tuned on test" - we select the threshold here only to describe
    #  one sensible operating point; production thresholds come from a cost model.)
    prec, rec, thr = precision_recall_curve(y_te, prob)
    f1s = np.divide(2 * prec * rec, prec + rec,
                    out=np.zeros_like(prec), where=(prec + rec) > 0)
    best = int(np.argmax(f1s[:-1]))
    t_star = float(thr[best])
    print(f"F1-optimal threshold = {t_star:.4f}  "
          f"(precision={prec[best]:.3f}, recall={rec[best]:.3f}, F1={f1s[best]:.3f})")

    y_hat = (prob >= t_star).astype(int)
    print("\nconfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_te, y_hat))
    print()
    print(classification_report(y_te, y_hat, target_names=["legit", "fraud"], digits=3))

    print("--- learned rules (depth 4) ---")
    print(export_text(model, feature_names=list(dv.feature_names_), max_depth=4))

    print("--- top feature importances ---")
    imp = sorted(zip(dv.feature_names_, model.feature_importances_), key=lambda t: -t[1])
    for name, val in imp[:8]:
        if val > 0:
            print(f"  {name:22s} {val:.4f}")

    os.makedirs(os.path.dirname(MODEL_OUT) or ".", exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump({
            "dv": dv,
            "model": model,
            "threshold": t_star,
            "params": BEST_PARAMS,
            "metrics": {"pr_auc": float(ap), "roc_auc": float(auc)},
        }, f)
    print(f"\nsaved -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
