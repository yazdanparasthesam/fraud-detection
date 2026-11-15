"""Streamlit UI for the PaySim fraud-scoring service.

The UI talks to the FastAPI backend over HTTP - it holds no model of its own.
That keeps a clean separation: predict.py owns inference, this file owns
presentation, and both can be scaled or redeployed independently.

    uvicorn predict:app --host 0.0.0.0 --port 9696     # backend
    streamlit run streamlit_app.py --server.port 8501  # this UI

Note the UI calls the API from Python (server-side), so `localhost:9696` is
correct here and works inside a container.

Two modes:
  * Single transaction - score one record by hand or from a preset.
  * Batch file        - upload CSV/Excel, score every row, download a report.
"""
from __future__ import annotations

import io
import os

import pandas as pd
import requests
import streamlit as st

from reasons import REQUIRED, detect_columns

API_URL = os.getenv("API_URL", "http://localhost:9696")

st.set_page_config(
    page_title="Fraud Detection - PaySim",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .verdict { padding: .9rem 1.1rem; border-radius: .6rem; font-size: 1.5rem;
             font-weight: 700; text-align: center; color: #fff; margin: .4rem 0; }
  .verdict-block { background: #b3261e; }
  .verdict-allow { background: #146c43; }
  .verdict-auto  { background: #4a5568; }
  .small-note { font-size: .82rem; color: #666; }
  .stMetric > div { font-size: .95rem; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- API helpers
@st.cache_data(ttl=15)
def health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_resource
def local_model():
    """Load the model in-process.

    Used when there is no API to call - notably on Streamlit Community Cloud,
    which runs a single container with no backend beside it. Returns None if the
    artefact is missing, so the UI can say so instead of crashing.
    """
    if os.getenv("DISABLE_LOCAL_MODEL") == "1":
        return None
    try:
        import inference
        return inference.get_model()
    except Exception:
        return None


def engine_label() -> str:
    return "FastAPI backend" if info else "in-process model"


def score(payload: dict) -> dict | None:
    if info:
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if r.status_code == 200:
                return r.json()
            st.error(f"API returned {r.status_code}: {r.text[:300]}")
        except Exception as exc:
            st.error(f"Cannot reach the API at {API_URL} — {type(exc).__name__}: {exc}")

    bundle = local_model()
    if bundle is None:
        st.error("No API reachable and no local model available.")
        return None
    try:
        import inference
        return inference.score_one(bundle, payload)
    except Exception as exc:
        st.error(f"Local scoring failed — {type(exc).__name__}: {exc}")
    return None


def score_file(csv_bytes: bytes, filename: str, include_legit: bool,
               limit: int = 50_000) -> dict | None:
    """Score an uploaded file - via the API if present, otherwise in-process."""
    if info:
        try:
            r = requests.post(
                f"{API_URL}/score-file",
                files={"file": (filename, csv_bytes, "text/csv")},
                params={"limit": limit, "include_legit": str(include_legit).lower()},
                timeout=600,
            )
            if r.status_code == 200:
                return r.json()
            detail = r.text[:400]
            try:
                detail = r.json().get("detail", detail)
            except Exception:
                pass
            st.error(f"API returned {r.status_code}: {detail}")
        except Exception as exc:
            st.error(f"Cannot reach the API at {API_URL} — {type(exc).__name__}: {exc}")

    bundle = local_model()
    if bundle is None:
        st.error("No API reachable and no local model available.")
        return None
    try:
        import inference
        import pandas as pd
        df = pd.read_csv(io.BytesIO(csv_bytes))
        return inference.score_frame(df, bundle, limit=limit,
                                     include_legit=include_legit,
                                     filename=filename)
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Local scoring failed — {type(exc).__name__}: {exc}")
    return None


def read_any(uploaded) -> pd.DataFrame:
    name = (uploaded.name or "").lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    return pd.read_csv(uploaded)


def to_excel(result: dict) -> bytes:
    """Build a two-sheet workbook: run summary + the returned rows."""
    rows = pd.DataFrame(result["rows"])
    s = result["summary"]
    summary = pd.DataFrame(
        [(k, v) for k, v in s.items() if k not in ("columns_detected", "warnings")]
        + [("columns_detected", str(s.get("columns_detected", {})))]
        + [("warnings", "; ".join(s.get("warnings", [])) or "none")]
        + [(f"evaluation.{k}", v) for k, v in s.get("evaluation", {}).items()],
        columns=["metric", "value"],
    )
    buf = io.BytesIO()
    engine = "xlsxwriter"
    try:
        import xlsxwriter  # noqa: F401
    except ImportError:
        engine = "openpyxl"
    with pd.ExcelWriter(buf, engine=engine) as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        rows.to_excel(xl, sheet_name="Results", index=False)
        if engine == "xlsxwriter":
            for sheet, frame in (("Summary", summary), ("Results", rows)):
                ws = xl.sheets[sheet]
                ws.freeze_panes(1, 0)
                for i, c in enumerate(frame.columns):
                    w = max(len(str(c)), 12)
                    ws.set_column(i, i, min(w + 2, 60))
    return buf.getvalue()


# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Service")
    st.code(f"API: {API_URL}", language="bash")
    info = health()
    if info:
        st.success("backend online")
        st.metric("test PR-AUC", f"{info['test_metrics']['pr_auc']:.4f}")
        st.metric("test ROC-AUC", f"{info['test_metrics']['roc_auc']:.4f}")
        st.metric("threshold", f"{info['threshold']:.4f}")
        st.caption(f"model: {info['model']}")
        st.caption(f"sha256: `{info['model_sha256'][:16]}…`")
        st.caption(f"scores: {', '.join(info['scored_types'])}")
    else:
        if local_model() is not None:
            st.warning("no backend — running the model in-process")
            st.caption(
                "This is the normal mode on Streamlit Community Cloud, which has "
                "no second container to host the API. Start the backend locally to "
                "use the HTTP service instead:"
            )
        else:
            st.error("backend offline and no local model found")
        st.code("uvicorn predict:app --host 0.0.0.0 --port 9696")

    st.divider()
    st.header("ℹ️ About")
    st.caption(
        "PaySim mobile-money fraud detection. ML Zoomcamp 2026 midterm project. "
        "A depth-4 decision tree scoring TRANSFER and CASH_OUT transactions; "
        "PAYMENT / CASH_IN / DEBIT are auto-approved because they carry zero fraud."
    )
    st.caption(
        "The dominant signal is `balance_after_send = oldbalanceOrg − amount`. "
        "Fraud drains the sender's account; legitimate transfers leave a balance."
    )
    st.caption(
        "Uploads only need the pre-transaction columns: step, type, amount, "
        "oldbalanceOrg, oldbalanceDest, nameDest. Post-transaction balances are "
        "ignored on purpose - they leak the label."
    )

# ------------------------------------------------------------------------ main
st.title("🛡️ Mobile-Money Fraud Detection")
st.caption("Score transactions against the deployed model. Requests go to the FastAPI service on port 9696.")

if not info and local_model() is not None:
    st.info("Running on the **in-process model** (no API configured). "
            "Start the backend to score over HTTP instead: "
            "`uvicorn predict:app --host 0.0.0.0 --port 9696`")
elif not info:
    st.warning("Start the backend before scoring: `uvicorn predict:app --host 0.0.0.0 --port 9696`")

tab_single, tab_batch = st.tabs(["🔍 Single transaction", "📂 Batch file"])

# =========================================================== TAB 1: single ====
with tab_single:
    st.subheader("Try an example")
    c1, c2, c3, c4 = st.columns(4)
    presets = {
        "p_fraud": dict(step=300, type="TRANSFER", amount=181.0, oldbalanceOrg=181.0,
                        oldbalanceDest=0.0, nameDest="C1234567890"),
        "p_legit": dict(step=300, type="TRANSFER", amount=1000.0, oldbalanceOrg=250000.0,
                        oldbalanceDest=5000.0, nameDest="C9876543210"),
        "p_cash": dict(step=420, type="CASH_OUT", amount=50000.0, oldbalanceOrg=50000.0,
                       oldbalanceDest=0.0, nameDest="M9988776655"),
        "p_pay": dict(step=300, type="PAYMENT", amount=5000.0, oldbalanceOrg=50000.0,
                      oldbalanceDest=0.0, nameDest="M1234567890"),
    }
    if c1.button("🔴 Fraudulent transfer", use_container_width=True,
                 help="TRANSFER that empties the account"):
        st.session_state.update(presets["p_fraud"])
    if c2.button("🟢 Legitimate transfer", use_container_width=True,
                 help="TRANSFER leaving most of the balance"):
        st.session_state.update(presets["p_legit"])
    if c3.button("🔴 Fraudulent cash-out", use_container_width=True,
                 help="CASH_OUT draining to a merchant"):
        st.session_state.update(presets["p_cash"])
    if c4.button("⚪ Payment (auto-approve)", use_container_width=True,
                 help="PAYMENT — zero fraud in this type"):
        st.session_state.update(presets["p_pay"])

    defaults = presets["p_fraud"]
    st.divider()
    st.subheader("Transaction details")

    types = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
    col1, col2 = st.columns(2)
    with col1:
        txn_type = st.selectbox(
            "type", types,
            index=types.index(st.session_state.get("type", defaults["type"])))
        amount = st.number_input("amount", min_value=0.0, step=100.0, format="%.2f",
                                 value=float(st.session_state.get("amount", defaults["amount"])))
        oldbalance_org = st.number_input(
            "oldbalanceOrg (sender balance before)", min_value=0.0, step=1000.0, format="%.2f",
            value=float(st.session_state.get("oldbalanceOrg", defaults["oldbalanceOrg"])))
    with col2:
        step = st.number_input("step (hour 1–744)", min_value=1, max_value=744, step=1,
                               value=int(st.session_state.get("step", defaults["step"])))
        oldbalance_dest = st.number_input(
            "oldbalanceDest (recipient balance before)", min_value=0.0, step=1000.0, format="%.2f",
            value=float(st.session_state.get("oldbalanceDest", defaults["oldbalanceDest"])))
        name_dest = st.text_input("nameDest (recipient id; 'M…' = merchant)",
                                  value=str(st.session_state.get("nameDest", defaults["nameDest"])))

    st.caption(
        "Post-transaction balances (`newbalanceOrig` / `newbalanceDest`) are deliberately not "
        "collected — in PaySim a flagged transfer is cancelled, so they leak the label."
    )

    score_clicked = st.button("🔍 Score transaction", type="primary", use_container_width=True)

    payload = {
        "step": int(step), "type": txn_type, "amount": float(amount),
        "oldbalanceOrg": float(oldbalance_org), "oldbalanceDest": float(oldbalance_dest),
        "nameDest": name_dest,
    }

    # live signal preview - the feature the model actually keys on
    balance_after = float(oldbalance_org) - float(amount)
    with st.expander("🔎 Why this transaction looks the way it does", expanded=False):
        st.write(
            f"**balance_after_send** = oldbalanceOrg − amount = "
            f"{oldbalance_org:,.2f} − {amount:,.2f} = **{balance_after:,.2f}**"
        )
        st.write(
            "The model's root split flags a transaction when this value falls in roughly "
            "**(−0.56, 0.37]** — i.e. the transfer empties the sender's account. "
            "This single feature carries **0.968** of the total importance."
        )
        st.progress(min(1.0, abs(balance_after) / max(oldbalance_org, 1.0)),
                    text="share of sender balance left after the transfer")

    if score_clicked:
        with st.spinner("Scoring…"):
            result = score(payload)

        if result:
            st.divider()
            st.subheader("Result")

            prob = result["fraud_probability"]
            decision = result["decision"]

            left, right = st.columns([1, 1.4])
            with left:
                st.metric("fraud probability", f"{prob:.4f}")
                st.progress(prob, text=f"{prob * 100:.2f}% risk")
                if decision == "block":
                    st.markdown('<div class="verdict verdict-block">🚫 BLOCK</div>',
                                unsafe_allow_html=True)
                elif decision == "auto_approve":
                    st.markdown('<div class="verdict verdict-auto">⚪ AUTO-APPROVE</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="verdict verdict-allow">✅ ALLOW</div>',
                                unsafe_allow_html=True)

                a, b = st.columns(2)
                a.metric("risk band", result.get("risk_band", "-"))
                b.metric("is_fraud", str(result.get("is_fraud")))
                if result.get("reason"):
                    st.info(result["reason"])

            with right:
                st.json(result)

            st.caption(f"decision threshold = {result.get('threshold')}")

# ============================================================ TAB 2: batch ====
with tab_batch:
    st.subheader("Score a whole file")
    st.caption(
        "Upload a CSV or Excel file of transactions. The service scores **every row**, "
        "returns the fraud cases it found, and explains why each one was flagged."
    )

    if not info:
        st.warning("The backend is offline — start it to score a file.")

    up = st.file_uploader("Choose a CSV or Excel file",
                          type=["csv", "xlsx", "xls"],
                          key="batch_upload")

    if up is not None:
        try:
            raw_df = read_any(up)
        except Exception as exc:
            st.error(f"Could not read that file — {type(exc).__name__}: {exc}")
            st.stop()

        st.success(f"Loaded **{len(raw_df):,} rows × {len(raw_df.columns)} columns** from `{up.name}`")

        detected = detect_columns(raw_df)
        missing_required = [c for c in REQUIRED if c not in detected]

        with st.expander("📋 Preview & column mapping", expanded=True):
            st.dataframe(raw_df.head(8), use_container_width=True)
            st.markdown(
                "These columns were recognised automatically. "
                "Override anything that looks wrong — `amount` and `oldbalanceOrg` are required."
            )
            cols = ["(not in file)"] + list(raw_df.columns)
            mapping = {}
            mcol1, mcol2 = st.columns(2)
            for i, canon in enumerate(["step", "type", "amount", "oldbalanceOrg",
                                       "oldbalanceDest", "nameDest", "isFraud"]):
                target = mcol1 if i % 2 == 0 else mcol2
                current = detected.get(canon, "(not in file)")
                idx = cols.index(current) if current in cols else 0
                mapping[canon] = target.selectbox(canon, cols, index=idx, key=f"map_{canon}")

        if missing_required:
            st.error(
                f"Missing required column(s): **{', '.join(missing_required)}**. "
                "Map them above before scoring."
            )

        include_legit = st.checkbox(
            "Also return rows that were cleared (not just the flagged ones)", value=False,
            help="Off by default so the report contains only suspected fraud.")

        go = st.button("🔍 Score file", type="primary",
                       disabled=bool(missing_required), use_container_width=True)

        if go:
            rev = {orig: canon for canon, orig in mapping.items() if orig != "(not in file)"}
            send_df = raw_df.rename(columns=rev)[list(rev.values())]
            csv_bytes = send_df.to_csv(index=False).encode("utf-8")
            with st.spinner(f"Scoring {len(raw_df):,} rows…"):
                st.session_state["batch_result"] = score_file(
                    csv_bytes, up.name, include_legit)

    result = st.session_state.get("batch_result")
    if result:
        s = result["summary"]
        rows = pd.DataFrame(result["rows"])

        st.divider()
        st.subheader("Report")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("rows scored", f"{s['scored_rows']:,}")
        k2.metric("fraud found", f"{s['flagged_fraud']:,}")
        k3.metric("flagged rate", f"{s['flagged_rate_pct']:.2f}%")
        k4.metric("auto-approved", f"{s['auto_approved_rows']:,}")

        if s.get("warnings"):
            for w in s["warnings"]:
                st.warning(w)
        if s.get("truncated"):
            st.info("The file was larger than the server's row cap; only part of it was scored.")

        ev = s.get("evaluation")
        if ev:
            st.success(
                f"Your file contained ground-truth labels — model on this file: "
                f"**precision {ev['precision']:.3f}**, **recall {ev['recall']:.3f}**, "
                f"**PR-AUC {ev['pr_auc']:.4f}**, **ROC-AUC {ev['roc_auc']:.4f}** "
                f"({ev['true_positives']} TP, {ev['false_positives']} FP, {ev['false_negatives']} FN)"
            )

        if rows.empty:
            st.info("No transactions were flagged. Tick ‘include cleared rows’ to inspect what the model cleared.")
        else:
            show = ["row", "type", "amount", "oldbalanceOrg", "oldbalanceDest",
                    "balance_after_send", "fraud_probability", "decision",
                    "risk_band", "why_flagged", "decision_rule"]
            show = [c for c in show if c in rows.columns]
            if "nameDest" in rows.columns:
                show.insert(4, "nameDest")
            if "actual_is_fraud" in rows.columns:
                show.append("actual_is_fraud")

            st.dataframe(rows[show], use_container_width=True, hide_index=True,
                         height=min(600, 80 + 34 * len(rows)))

            st.caption(
                "`why_flagged` names the condition that pinned the row down; "
                "`decision_rule` is the full path through the tree."
            )

            d1, d2 = st.columns(2)
            d1.download_button(
                "⬇️ Download CSV", rows[show].to_csv(index=False).encode("utf-8"),
                "fraud_report.csv", "text/csv", use_container_width=True)
            d2.download_button(
                "⬇️ Download Excel", to_excel({"summary": s, "rows": rows[show]}),
                "fraud_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)
