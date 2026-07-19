from __future__ import annotations

import numpy as np
import pandas as pd

from finauth_audit.evaluation.metrics import with_decision_outcomes


def with_laundering_outcomes(frame: pd.DataFrame, decision: np.ndarray) -> pd.DataFrame:
    result = with_decision_outcomes(frame, decision)
    result["selected_direct_leakage"] = result["authorized"] & ~result["current_source_eligible"]
    result["selected_indirect_leakage"] = result["authorized"] & result["authority_laundering"]
    result["positive_safe_delegation"] = result["safe_delegation"] & (result["best_safe_utility"] > 0)
    result["false_block"] = result["positive_safe_delegation"] & ~result["authorized"]
    result["laundering_harm"] = result["selected_indirect_leakage"] & result["selected_harm"]
    return result


def laundering_metrics(frame: pd.DataFrame) -> dict[str, object]:
    n = len(frame)
    if n == 0:
        raise ValueError("empty provenance profile")
    authorized = int(frame["authorized"].sum())
    safe_delegation = int(frame["positive_safe_delegation"].sum())
    laundering_executions = int(frame["selected_indirect_leakage"].sum())
    return {
        "rows": n,
        "authorized_count": authorized,
        "coverage": authorized / n,
        "far": float(frame.loc[frame["authorized"], "selected_harm"].mean()) if authorized else None,
        "alr": float(frame.loc[frame["authorized"], "selected_laundering"].mean()) if authorized else None,
        "direct_leakage_rate": float(
            frame.loc[frame["authorized"], "selected_direct_leakage"].mean()
        )
        if authorized
        else None,
        "indirect_leakage_rate": float(
            frame.loc[frame["authorized"], "selected_indirect_leakage"].mean()
        )
        if authorized
        else None,
        "authority_laundering_count": laundering_executions,
        "laundering_harm_rate": float(
            frame.loc[frame["selected_indirect_leakage"], "selected_harm"].mean()
        )
        if laundering_executions
        else None,
        "safe_delegation_count": safe_delegation,
        "safe_delegation_coverage": float(
            frame.loc[frame["positive_safe_delegation"], "authorized"].mean()
        )
        if safe_delegation
        else None,
        "false_block_rate": float(frame.loc[frame["positive_safe_delegation"], "false_block"].mean())
        if safe_delegation
        else None,
        "cau": float(frame.loc[frame["authorized"], "selected_utility"].sum()) if authorized else 0.0,
        "moc": float(frame["missed_opportunity"].sum()),
        "review_rate": float(frame["decision"].eq("review").mean()),
        "abstain_rate": float(frame["decision"].eq("abstain").mean()),
        "zero_action": authorized == 0,
    }
