"""Score every transaction with BOTH detectors and export an Excel report.

Two detectors, deliberately different in kind:

1. SUPERVISED  - DecisionTreeClassifier (the deployed model).
   Learns from the `isFraud` label. Produces a probability and, because it is a
   depth-4 tree, a fully traceable rule for every prediction.

2. UNSUPERVISED - IsolationForest.
   Never sees the label. Flags transactions that are easy to isolate in feature
   space. Catches *novel* patterns the tree was never trained on - at the cost
   of many more false positives.

Every flagged row carries a REASON:
  - tree: the exact decision-path rule that fired, plus the single field that
          drove the outcome (highest-importance feature on the path).
  - forest: the two fields furthest from the population median, as robust
          z-scores against the interquartile range.

Output: outputs/fraud_report.xlsx
"""
from __future__ import annotations

import os
import pickle
import gc
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

from reasons import build_leaf_paths, humanise, pick_trigger, readable_rule
from vectorize import transform_chunked

MODEL_PATH = "models/fraud_model.bin"
DATA = "data/paysim_model.parquet"
OUT_DIR = "outputs"
OUT_XLSX = os.path.join(OUT_DIR, "fraud_report.xlsx")

CHUNK = 250_000
CONTAMINATION = 0.01      # IsolationForest: expected share of outliers
SAMPLE_ROWS = 25_000      # rows in the "Scored Sample" sheet
RANDOM_STATE = 42

# Plain-English meaning for the fields most often behind a flag
FIELD_NOTES = {
    "balance_after_send": "sender balance minus amount - ~0 means the transfer empties the account",
    "amount_to_balance": "amount divided by sender balance - huge ratios mean 'sending more than they have'",
    "amount": "transaction size",
    "oldbalanceOrg": "sender balance BEFORE the transaction",
    "oldbalanceDest": "recipient balance BEFORE the transaction",
    "log_amount": "log(1 + amount), used to tame the long right tail",
    "log_oldbalance_org": "log(1 + oldbalanceOrg)",
    "step": "hour since the start of the log (1-744)",
    "hour_of_day": "step mod 24",
    "day": "step div 24",
    "orig_balance_zero": "sender balance was exactly zero",
    "dest_balance_zero": "recipient balance was exactly zero",
    "dest_is_merchant": "recipient id starts with 'M'",
    "amount_zero": "amount was exactly zero",
    "type=CASH_OUT": "transaction is a CASH_OUT",
    "type=TRANSFER": "transaction is a TRANSFER",
}


# --------------------------------------------------------------------------- #
# Decision-tree reason codes
# --------------------------------------------------------------------------- #
def build_leaf_rules(model, feature_names):
    """Return {leaf_id: (list of (field, op, threshold), readable_rule)}.

    With max_depth=4 there are only 13 leaves, so we resolve the rule for each
    leaf ONCE and reuse it for millions of rows.
    """
    t = model.tree_
    parent = {}
    for node in range(t.node_count):
        left, right = t.children_left[node], t.children_right[node]
        if left != -1:
            parent[left] = (node, "left")
            parent[right] = (node, "right")

    importance = dict(zip(feature_names, model.feature_importances_))

    def conditions(leaf):
        conds, node = [], leaf
        while node in parent:
            p, side = parent[node]
            field = feature_names[t.feature[p]]
            thr = float(t.threshold[p])
            conds.append((field, "<=" if side == "left" else ">", thr))
            node = p
        return list(reversed(conds))

    def readable(conds):
        parts = []
        for f, op, thr in conds:
            if f in ("type=CASH_OUT", "type=TRANSFER"):
                parts.append(f"type {'is' if op == '>' else 'is not'} {f.split('=')[1]}")
            else:
                parts.append(f"{f} {op} {thr:,.2f}")
        return " AND ".join(parts)

    rules = {}
    for leaf in range(t.node_count):
        if t.children_left[leaf] == -1 and t.children_right[leaf] == -1:
            c = conditions(leaf)
            if not c:
                continue
            rules[leaf] = (c, readable(c))
    return rules


def robust_z_stats(X: np.ndarray):
    med = np.median(X, axis=0)
    q1, q3 = np.percentile(X, [25, 75], axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1.0
    return med, iqr


def top_anomalous_fields(X, med, iqr, k=2):
    """For each row, the k fields furthest from the population median.

    Returns (indices, signs, z-values) as compact arrays so we don't hold a
    full 2.77M x 16 z-score matrix in memory.
    """
    n = X.shape[0]
    idx = np.zeros((n, k), dtype=np.int8)
    sgn = np.zeros((n, k), dtype=np.int8)
    val = np.zeros((n, k), dtype=np.float32)
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        z = (X[start:stop] - med) / iqr
        # rank fields by |z| descending
        order = np.argsort(-np.abs(z), axis=1, kind="stable")[:, :k]
        rows = np.arange(stop - start)[:, None]
        idx[start:stop] = order
        sgn[start:stop] = np.sign(z[rows, order])
        val[start:stop] = np.abs(z[rows, order])
    return idx, sgn, val


# --------------------------------------------------------------------------- #
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    df = pd.read_parquet(DATA)
    y = df["isFraud"].values.astype("int8")
    Xdf = df[[c for c in df.columns if c != "isFraud"]]
    n = len(df)
    print(f"loaded {n:,} transactions ({y.sum():,} fraudulent)")

    with open(MODEL_PATH, "rb") as f:
        art = pickle.load(f)
    dv, tree = art["dv"], art["model"]
    threshold = art["threshold"]
    feature_names = list(dv.feature_names_)

    # ---------------- design matrix (chunked, float32) ----------------
    print("vectorising…")
    X = np.empty((n, len(feature_names)), dtype=np.float32)
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        X[start:stop] = transform_chunked(
            dv, Xdf.iloc[start:stop]).toarray().astype(np.float32)
    print(f"  X = {X.shape}  ({X.nbytes / 1e6:.0f} MB)")

    # ---------------- 1. supervised decision tree ----------------
    print("scoring with the decision tree…")
    prob = tree.predict_proba(X)[:, 1]
    leaf = tree.apply(X)
    rules = build_leaf_rules(tree, feature_names)
    print(f"  resolved {len(rules)} distinct leaf rules")

    # ---------------- 2. unsupervised isolation forest ----------------
    print("training + scoring IsolationForest…")
    iso = IsolationForest(n_estimators=100, contamination=CONTAMINATION,
                          random_state=RANDOM_STATE, n_jobs=1)
    iso.fit(X)
    if_score = iso.decision_function(X)      # higher = more normal
    if_flag = iso.predict(X) == -1           # -1 == anomaly
    print(f"  flagged {if_flag.sum():,} anomalies ({100 * if_flag.mean():.2f}%)")

    # ---------------- reasons ----------------
    print("deriving reason codes…")
    med, iqr = robust_z_stats(X)
    f_idx, f_sgn, f_val = top_anomalous_fields(X, med, iqr, k=2)

    # Which condition on a leaf's path actually made this row unusual?
    # Global feature importance always names the root split, which every row
    # shares. Instead we measure how rare each condition is across the whole
    # population: the rarer the constraint the row satisfies, the more that
    # specific field is what pinned the transaction down.
    print("  measuring rarity of each path condition…")
    support = {}
    for conds, _ in rules.values():
        for cond in conds:
            if cond not in support:
                field, op, thr = cond
                col = X[:, feature_names.index(field)]
                support[cond] = float(np.mean(col <= thr) if op == "<=" else np.mean(col > thr))

    rule_by_leaf, driver_by_leaf, human_by_leaf, rarity_by_leaf = {}, {}, {}, {}
    for lf, (conds, readable_rule) in rules.items():
        rule_by_leaf[lf] = readable_rule
        cond = min(conds, key=lambda c: support[c])          # rarest constraint
        driver_by_leaf[lf] = cond[0]
        human_by_leaf[lf] = humanise(*cond)
        rarity_by_leaf[lf] = round(100 * support[cond], 3)

    # X is no longer needed - release 177 MB before building the report frame
    del X
    gc.collect()


    dt_flag = prob >= threshold

    # ---------------- assemble (memory-light) ----------------
    # We never build reason strings for all 2.77M rows - only for the rows that
    # actually land in a sheet. That keeps peak memory around 400 MB instead of
    # several GB.
    print("assembling report…")
    dt_flag = prob >= threshold
    rng = np.random.default_rng(RANDOM_STATE)

    base = pd.DataFrame({
        "txn_id": np.arange(n, dtype=np.int32),
        "step": Xdf["step"].values,
        "type": Xdf["type"].values,
        "amount": Xdf["amount"].values,
        "oldbalanceOrg": Xdf["oldbalanceOrg"].values,
        "oldbalanceDest": Xdf["oldbalanceDest"].values,
        "hour_of_day": Xdf["hour_of_day"].values,
        "day": Xdf["day"].values,
        "balance_after_send": Xdf["balance_after_send"].values,
        "amount_to_balance": Xdf["amount_to_balance"].values,
        "dest_is_merchant": Xdf["dest_is_merchant"].values,
        "dt_probability": prob.astype("float32"),
        "dt_flagged": dt_flag,
        "if_anomaly_score": if_score.astype("float32"),
        "if_flagged": if_flag,
        "actual_is_fraud": y,
    })

    def if_reason(i):
        parts = []
        for j in range(2):
            field = feature_names[f_idx[i, j]]
            direction = "HIGH" if f_sgn[i, j] > 0 else "LOW"
            parts.append(f"unusually {direction} {field} (z={f_val[i, j]:,.1f})")
        return "; ".join(parts)

    def build(idx: np.ndarray) -> pd.DataFrame:
        """Slice rows and attach reason strings for just those rows."""
        f = base.iloc[idx].copy()
        d_ = f.dt_flagged.values          # did the tree flag it?
        i_ = f.if_flagged.values          # did the forest flag it?

        # Only report a reason for the detector that actually fired - otherwise
        # a row the tree cleared would still carry a tree "reason".
        f["dt_rule"] = [rule_by_leaf.get(leaf[i], "") if d_ else "" for i, d_ in zip(idx, d_)]
        f["dt_trigger_field"] = [driver_by_leaf.get(leaf[i], "") if d_ else ""
                                 for i, d_ in zip(idx, d_)]
        f["dt_trigger_rarity_pct"] = [rarity_by_leaf.get(leaf[i], np.nan) if d_ else np.nan
                                      for i, d_ in zip(idx, d_)]
        f["dt_reason"] = [human_by_leaf.get(leaf[i], "") if d_ else ""
                          for i, d_ in zip(idx, d_)]
        f["if_top_field"] = [feature_names[f_idx[i, 0]] if flagged else ""
                             for i, flagged in zip(idx, i_)]
        f["if_reason"] = [if_reason(i) if i_ else "" for i, i_ in zip(idx, i_)]

        f["models_agree"] = d_ == i_
        f["detection"] = np.where(d_ & i_, "both",
                                  np.where(d_, "tree only",
                                           np.where(i_, "forest only", "neither")))
        return f

    m_susp = dt_flag | if_flag
    m_conf = y == 1
    m_dis = dt_flag != if_flag

    idx_susp = np.where(m_susp)[0]
    idx_conf = np.where(m_conf)[0]
    idx_dis = np.where(m_dis)[0]
    idx_samp = rng.choice(n, size=min(SAMPLE_ROWS, n), replace=False)
    idx_samp.sort()

    suspected = build(idx_susp)
    confirmed = build(idx_conf)
    disagreements = build(idx_dis)
    sample = build(idx_samp)
    del base
    gc.collect()

    # ---------------- metrics ----------------
    dt_ap = average_precision_score(y, prob)
    dt_auc = roc_auc_score(y, prob)
    if_ap = average_precision_score(y, -if_score)      # lower score = more anomalous
    if_auc = roc_auc_score(y, -if_score)

    dt_tp = int((dt_flag & (y == 1)).sum())
    if_tp = int((if_flag & (y == 1)).sum())
    dt_fp = int((dt_flag & (y == 0)).sum())
    if_fp = int((if_flag & (y == 0)).sum())
    n_fraud = int(y.sum())

    comparison = pd.DataFrame([
        {"model": "Decision Tree (supervised)", "uses_labels": "yes",
         "flagged": int(dt_flag.sum()), "true_positives": dt_tp, "false_positives": dt_fp,
         "precision": round(dt_tp / max(dt_tp + dt_fp, 1), 4),
         "recall": round(dt_tp / max(n_fraud, 1), 4),
         "PR_AUC": round(dt_ap, 4), "ROC_AUC": round(dt_auc, 4)},
        {"model": "Isolation Forest (unsupervised)", "uses_labels": "no",
         "flagged": int(if_flag.sum()), "true_positives": if_tp, "false_positives": if_fp,
         "precision": round(if_tp / max(if_tp + if_fp, 1), 4),
         "recall": round(if_tp / max(n_fraud, 1), 4),
         "PR_AUC": round(if_ap, 4), "ROC_AUC": round(if_auc, 4)},
    ])
    print()
    print(comparison.to_string(index=False))

    # ---------------- Excel ----------------
    print(f"\nwriting {OUT_XLSX} …")
    summary = pd.DataFrame([
        ("Transactions scored", f"{n:,}"),
        ("Fraudulent (ground truth)", f"{n_fraud:,} ({100*y.mean():.4f}%)"),
        ("", ""),
        ("Decision Tree flagged", f"{int(dt_flag.sum()):,}"),
        ("  of which truly fraud", f"{dt_tp:,}"),
        ("  false positives", f"{dt_fp:,}"),
        ("  PR-AUC / ROC-AUC", f"{dt_ap:.4f} / {dt_auc:.4f}"),
        ("  NOTE", "above is over ALL rows (incl. training). Held-out "
                   "test-set PR-AUC = 0.9968"),
        ("", ""),
        ("Isolation Forest flagged", f"{int(if_flag.sum()):,}"),
        ("  of which truly fraud", f"{if_tp:,}"),
        ("  false positives", f"{if_fp:,}"),
        ("  PR-AUC / ROC-AUC", f"{if_ap:.4f} / {if_auc:.4f}"),
        ("  contamination setting", f"{CONTAMINATION}"),
        ("", ""),
        ("Flagged by BOTH", f"{int((dt_flag & if_flag).sum()):,}"),
        ("Tree only", f"{int((dt_flag & ~if_flag).sum()):,}"),
        ("Forest only", f"{int((~dt_flag & if_flag).sum()):,}"),
        ("", ""),
        ("Rows in 'Suspected Fraud' sheet", f"{len(suspected):,}"),
        ("Rows in 'Confirmed Fraud' sheet", f"{len(confirmed):,}"),
        ("Rows in 'Disagreements' sheet", f"{len(disagreements):,}"),
        ("Rows in 'Scored Sample' sheet", f"{len(sample):,}"),
        ("", ""),
        ("Decision threshold (tree)", f"{threshold:.6f}"),
        ("Generated", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")),
    ], columns=["metric", "value"])

    reference = pd.DataFrame(sorted(FIELD_NOTES.items()),
                             columns=["field / flag", "meaning"])
    reference = pd.concat([reference, pd.DataFrame([
        ("dt_probability", "model's estimated probability of fraud"),
        ("dt_flagged", "probability >= decision threshold"),
        ("dt_rule", "the exact decision-path rule that fired"),
        ("dt_trigger_field", "the field that actually pinned this row down"),
        ("dt_trigger_rarity_pct", "% of all transactions satisfying that condition - lower = rarer"),
        ("dt_reason", "plain-English version of the driver condition"),
        ("if_anomaly_score", "IsolationForest score - LOWER means more anomalous"),
        ("if_flagged", "IsolationForest labelled the row an anomaly"),
        ("if_top_field", "field furthest from the population median"),
        ("if_reason", "two fields furthest from the median (robust z vs IQR)"),
        ("detection", "both / tree only / forest only / neither"),
    ], columns=["field / flag", "meaning"])], ignore_index=True)

    sheets = [("Summary", summary), ("Model Comparison", comparison),
              ("Suspected Fraud", suspected), ("Confirmed Fraud", confirmed),
              ("Disagreements", disagreements), ("Scored Sample", sample),
              ("Field Reference", reference)]

    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except ImportError:
        engine = "openpyxl"
    print(f"  engine: {engine}")

    with pd.ExcelWriter(OUT_XLSX, engine=engine) as xl:
        for name, frame in sheets:
            frame.to_excel(xl, sheet_name=name, index=False)

        for name, frame in sheets:
            head = frame.head(400)
            if engine == "xlsxwriter":
                ws = xl.sheets[name]
                ws.freeze_panes(1, 0)
                for i, col in enumerate(frame.columns):
                    w = max(len(str(col)),
                            int(head[col].astype(str).str.len().max()) if len(head) else 10)
                    ws.set_column(i, i, min(max(w + 2, 11), 60))
            else:
                ws = xl.sheets[name]
                ws.freeze_panes = "A2"
                for i, col in enumerate(frame.columns, start=1):
                    w = max(len(str(col)),
                            int(head[col].astype(str).str.len().max()) if len(head) else 10)
                    ws.column_dimensions[
                        ws.cell(row=1, column=i).column_letter].width = min(max(w + 2, 11), 60)

    print(f"  wrote {OUT_XLSX}  ({os.path.getsize(OUT_XLSX)/1e6:.1f} MB)")
    print(f"\ndone in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
