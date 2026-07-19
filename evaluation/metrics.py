from __future__ import annotations

import math

import numpy as np
import pandas as pd


AUTHORIZED = {"execute", "reduce"}


def with_decision_outcomes(frame: pd.DataFrame, decision: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    result["decision"] = decision
    result["authorized"] = result["decision"].isin(AUTHORIZED)
    result["selected_utility"] = np.select(
        [result["decision"].eq("execute"), result["decision"].eq("reduce")],
        [result["full_utility"], result["reduced_utility"]],
        default=0.0,
    )
    result["selected_harm"] = result["authorized"] & (result["selected_utility"] < 0)
    result["selected_laundering"] = result["authorized"] & ~result["original_source_eligible"]
    result["best_safe_utility"] = np.maximum.reduce(
        [result["full_utility"].to_numpy(), result["reduced_utility"].to_numpy(), np.zeros(len(result))]
    )
    result["missed_opportunity"] = np.where(
        ~result["authorized"] & result["original_source_eligible"],
        result["best_safe_utility"],
        0.0,
    )
    return result


def profile_metrics(frame: pd.DataFrame) -> dict[str, object]:
    n = len(frame)
    if n == 0:
        raise ValueError("profile is empty")
    authorized = int(frame["authorized"].sum())
    selected = frame.loc[frame["authorized"], "selected_utility"]
    direct_q05 = float(frame["full_utility"].quantile(0.05))
    selected_q05 = float(frame["selected_utility"].quantile(0.05))
    return {
        "rows": n,
        "authorized_count": authorized,
        "coverage": authorized / n,
        "execute_rate": float(frame["decision"].eq("execute").mean()),
        "reduce_rate": float(frame["decision"].eq("reduce").mean()),
        "review_rate": float(frame["decision"].eq("review").mean()),
        "abstain_rate": float(frame["decision"].eq("abstain").mean()),
        "far": float(frame.loc[frame["authorized"], "selected_harm"].mean()) if authorized else None,
        "alr": float(frame.loc[frame["authorized"], "selected_laundering"].mean()) if authorized else None,
        "cau": float(selected.sum()) if authorized else 0.0,
        "upa": float(selected.mean()) if authorized else None,
        "moc": float(frame["missed_opportunity"].sum()),
        "tla": selected_q05 - direct_q05,
        "zero_action": authorized == 0,
        "n_a_execution_metric": authorized == 0,
    }


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        return None
    return numerator / denominator
