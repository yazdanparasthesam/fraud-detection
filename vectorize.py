"""Memory-bounded DictVectorizer transform.

`to_dict(orient='records')` on millions of rows materialises millions of Python
dicts and blows up memory, so we transform in chunks and stack the result.
Keeps train/serve encoding identical, since predict.py uses the same fitted
DictVectorizer.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse


def transform_chunked(dv, frame, chunk: int = 50_000):
    """Transform a DataFrame with a fitted DictVectorizer, one chunk at a time."""
    parts = []
    n = len(frame)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = frame.iloc[start:stop].to_dict(orient="records")
        parts.append(dv.transform(block))
        del block
    if not parts:
        return sparse.csr_matrix((0, len(dv.feature_names_)))
    return sparse.vstack(parts).tocsr()


def fit_chunked(dv, frame, chunk: int = 50_000):
    """Fit on the first chunk (categorical levels are few), then transform all."""
    head = frame.iloc[:chunk].to_dict(orient="records")
    dv.fit(head)
    return transform_chunked(dv, frame, chunk)
