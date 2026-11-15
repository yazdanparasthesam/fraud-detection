"""Shared reason-code helpers.

Used by both the online service (`predict.py`) and the offline report
(`score_and_export.py`) so a transaction is explained the same way everywhere.

A depth-4 decision tree has only 13 leaves, so every prediction can be traced
to an exact list of split conditions. We resolve those rules once per tree and
reuse them for millions of rows.
"""
from __future__ import annotations


def build_leaf_paths(model, feature_names: list[str]) -> dict:
    """{leaf_id: [(field, op, threshold), ...]} walking root -> leaf."""
    t = model.tree_
    parent = {}
    for node in range(t.node_count):
        left, right = t.children_left[node], t.children_right[node]
        if left != -1:
            parent[left] = (node, "left")
            parent[right] = (node, "right")

    paths = {}
    for leaf in range(t.node_count):
        if t.children_left[leaf] == -1 and t.children_right[leaf] == -1:
            conds, node = [], leaf
            while node in parent:
                p, side = parent[node]
                field = feature_names[t.feature[p]]
                conds.append((field, "<=" if side == "left" else ">",
                              float(t.threshold[p])))
                node = p
            conds.reverse()
            if conds:
                paths[leaf] = conds
    return paths


def readable_rule(conds) -> str:
    """Plain 'field op threshold AND ...' rendering of a decision path."""
    parts = []
    for f, op, thr in conds:
        if f in ("type=CASH_OUT", "type=TRANSFER"):
            parts.append(f"type {'is' if op == '>' else 'is not'} {f.split('=')[1]}")
        else:
            parts.append(f"{f} {op} {thr:,.2f}")
    return " AND ".join(parts)


def pick_trigger(conds, support: dict | None = None):
    """Choose the condition on a path that best explains *this* row.

    With population `support` (fraction of rows satisfying each condition) we
    pick the RAREST constraint - global feature importance would just name the
    root split that every row shares, which explains nothing.

    Without it we fall back to the deepest condition on the path, which is the
    most specific test this row actually passed.
    """
    if support:
        return min(conds, key=lambda c: support.get(c, 1.0))
    return conds[-1]


def humanise(field: str, op: str, thr: float) -> str:
    """Turn one split condition into plain English."""
    if field == "balance_after_send" and op == "<=" and abs(thr) < 1:
        return ("sender's balance is emptied by the transfer "
                f"(balance after sending <= {thr:,.2f})")
    if field == "balance_after_send" and op == ">":
        return f"sender keeps money behind (balance after sending > {thr:,.2f})"
    if field == "amount_to_balance" and op == ">":
        return f"amount is more than {thr:,.1f}x the sender's balance"
    if field == "amount" and op == ">":
        return f"unusually large amount (> {thr:,.0f})"
    if field == "oldbalanceDest" and op == "<=":
        return f"recipient account was nearly empty (<= {thr:,.0f})"
    if field == "oldbalanceDest" and op == ">":
        return f"recipient held a large balance (> {thr:,.0f})"
    if field == "step" and op == "<=":
        return f"early in the observation window (step <= {thr:,.0f})"
    if field == "step" and op == ">":
        return f"later in the observation window (step > {thr:,.0f})"
    if field == "log_amount" and op == ">":
        return f"large amount (log1p > {thr:,.2f})"
    if field == "log_oldbalance_org" and op == "<=":
        return f"sender balance was tiny (log1p <= {thr:,.2f})"
    if field == "dest_balance_zero" and op == "<=":
        return "recipient account held a non-zero balance"
    if field == "dest_balance_zero" and op == ">":
        return "recipient balance was exactly zero"
    if field == "oldbalanceDest" and op == "<=":
        return f"recipient balance was low (<= {thr:,.0f})"
    if field == "oldbalanceDest" and op == ">":
        return f"recipient held a large balance (> {thr:,.0f})"
    if field == "oldbalanceOrg" and op == "<=":
        return f"sender balance was low (<= {thr:,.0f})"
    if field == "oldbalanceOrg" and op == ">":
        return f"sender held a large balance (> {thr:,.0f})"
    if field == "orig_balance_zero" and op == ">":
        return "sender balance was exactly zero"
    if field == "hour_of_day" and op == "<=":
        return f"transaction happened in the first hours of the day (<= {thr:,.0f}h)"
    if field == "hour_of_day" and op == ">":
        return f"transaction happened late in the day (> {thr:,.0f}h)"
    if field == "type=CASH_OUT" and op == ">":
        return "transaction is a CASH_OUT"
    if field == "type=TRANSFER" and op == ">":
        return "transaction is a TRANSFER"
    return f"{field} {op} {thr:,.2f}"


# --------------------------------------------------------------------------- #
# IsolationForest reason codes
# --------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------
# Column detection for uploaded files
#
# Shared by the API and the Streamlit UI so both agree on what a column means.
# Live here (not in predict.py) because importing predict.py would load the
# model into the UI process.
# ---------------------------------------------------------------------------
ALIASES = {
    "step":           ["step", "hour", "time", "timestep"],
    "type":           ["type", "txtype", "transactiontype", "paymenttype"],
    "amount":         ["amount", "amt", "value", "transactionamount",
                       "txnamount", "transactionamt"],
    "nameOrig":       ["nameorig", "sender", "origin", "fromaccount"],
    "oldbalanceOrg":  ["oldbalanceorg", "oldbalanceorig", "senderbalance",
                       "origbalance", "balanceorig", "openingbalance",
                       "senderbalancebefore", "frombalance"],
    "oldbalanceDest": ["oldbalancedest", "destbalance", "balancedest",
                       "recipientbalance", "destinationbalance"],
    "nameDest":       ["namedest", "destination", "recipient", "toaccount",
                       "recipientid", "destid", "beneficiary"],
    "isFraud":        ["isfraud", "fraud", "label", "class", "target",
                       "fraudlabel", "fraudflag", "isfraudulent", "actualfraud"],
}
REQUIRED = ["amount", "oldbalanceOrg"]
OPTIONAL = ["step", "type", "nameOrig", "oldbalanceDest", "nameDest", "isFraud"]


def norm(name: str) -> str:
    """Lowercase and strip separators so 'Old Balance Org' == 'oldbalanceOrg'."""
    import re
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def detect_columns(df) -> dict:
    """Map canonical field -> actual column name found in the uploaded frame."""
    lookup = {norm(c): c for c in df.columns}
    found = {}
    for canon, alts in ALIASES.items():
        for a in alts:
            if a in lookup:
                found[canon] = lookup[a]
                break
    return found
