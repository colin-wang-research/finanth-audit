from __future__ import annotations

import itertools

import pandas as pd


# The 6 x 4 x 5 grid is frozen before test access. Bounds are estimated once
# per rule/profile; these cells are deterministic operating constraints, not
# 120 independent hypothesis tests.
HARM_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
LAUNDERING_THRESHOLDS = (0.00, 0.01, 0.02, 0.05)
MIN_COVERAGE = (0.05, 0.10, 0.20, 0.30, 0.50)


def build_surface(bounds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for record in bounds.to_dict(orient="records"):
        for tau_h, tau_a, c_min in itertools.product(
            HARM_THRESHOLDS, LAUNDERING_THRESHOLDS, MIN_COVERAGE
        ):
            required = (record["far_ucb95"], record["alr_ucb95"], record["coverage_lcb95"])
            passed = all(pd.notna(value) for value in required)
            if passed:
                passed = bool(
                    record["far_ucb95"] <= tau_h
                    and record["alr_ucb95"] <= tau_a
                    and record["coverage_lcb95"] >= c_min
                )
            rows.append(
                {
                    "rule": record["rule"],
                    "profile": record["profile"],
                    "tau_h": tau_h,
                    "tau_a": tau_a,
                    "c_min": c_min,
                    "passed": passed,
                    "failure_far": pd.isna(record["far_ucb95"]) or record["far_ucb95"] > tau_h,
                    "failure_alr": pd.isna(record["alr_ucb95"]) or record["alr_ucb95"] > tau_a,
                    "failure_coverage": record["coverage_lcb95"] < c_min,
                }
            )
    surface = pd.DataFrame(rows)
    by_profile = (
        surface.groupby(["rule", "profile"], as_index=False)["passed"]
        .mean()
        .rename(columns={"passed": "certification_volume"})
    )
    pivot = by_profile.pivot(index="rule", columns="profile", values="certification_volume")
    summary = pivot.reset_index()
    scored_profiles = [column for column in ("overall", "stress") if column in summary.columns]
    summary["worst_profile_certification_volume"] = summary[scored_profiles].min(axis=1)
    return surface, summary
