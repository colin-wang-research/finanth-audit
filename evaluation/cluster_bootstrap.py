from __future__ import annotations

import numpy as np
import pandas as pd


def _quantile(values: list[float], q: float) -> float | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return float(np.quantile(finite, q)) if len(finite) else None


def cluster_bootstrap_bounds(
    frame: pd.DataFrame,
    replicates: int,
    seed: int,
    chunk_size: int = 64,
) -> dict[str, float | int | None]:
    grouped = (
        frame.assign(
            harmful=frame["selected_harm"].astype(int),
            laundering=frame["selected_laundering"].astype(int),
            authorized_int=frame["authorized"].astype(int),
        )
        .groupby("event_cluster_id", sort=False)
        .agg(
            rows=("row_id", "size"),
            authorized=("authorized_int", "sum"),
            harmful=("harmful", "sum"),
            laundering=("laundering", "sum"),
            utility=("selected_utility", "sum"),
        )
    )
    if grouped.empty:
        raise ValueError("no event clusters")

    arrays = {column: grouped[column].to_numpy(dtype=float) for column in grouped.columns}
    cluster_count = len(grouped)
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    rng = np.random.default_rng(seed)
    coverages: list[float] = []
    fars: list[float] = []
    alrs: list[float] = []
    caus: list[float] = []
    upas: list[float] = []
    zero_authorized_replicates = 0

    for start in range(0, replicates, chunk_size):
        batch = min(chunk_size, replicates - start)
        sample = rng.integers(0, cluster_count, size=(batch, cluster_count))
        rows = arrays["rows"][sample].sum(axis=1)
        authorized = arrays["authorized"][sample].sum(axis=1)
        utilities = arrays["utility"][sample].sum(axis=1)
        coverages.extend((authorized / rows).astype(float).tolist())
        caus.extend(utilities.astype(float).tolist())
        active = authorized > 0
        zero_authorized_replicates += int((~active).sum())
        if active.any():
            harmful = arrays["harmful"][sample].sum(axis=1)
            laundering = arrays["laundering"][sample].sum(axis=1)
            fars.extend((harmful[active] / authorized[active]).astype(float).tolist())
            alrs.extend((laundering[active] / authorized[active]).astype(float).tolist())
            upas.extend((utilities[active] / authorized[active]).astype(float).tolist())

    return {
        "clusters": cluster_count,
        "replicates": replicates,
        "zero_authorized_replicates": zero_authorized_replicates,
        "coverage_lcb95": _quantile(coverages, 0.05),
        "coverage_ucb95": _quantile(coverages, 0.95),
        "far_ucb95": _quantile(fars, 0.95),
        "alr_ucb95": _quantile(alrs, 0.95),
        "cau_lcb95": _quantile(caus, 0.05),
        "cau_ucb95": _quantile(caus, 0.95),
        "upa_lcb95": _quantile(upas, 0.05),
        "upa_ucb95": _quantile(upas, 0.95),
    }
