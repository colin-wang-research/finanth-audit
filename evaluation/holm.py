from __future__ import annotations

from collections.abc import Mapping


def holm_adjust(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict[str, float | bool]]:
    ordered = sorted((float(value), str(name)) for name, value in pvalues.items())
    total = len(ordered)
    adjusted_running = 0.0
    rejected_prefix = True
    output: dict[str, dict[str, float | bool]] = {}
    for rank, (raw, name) in enumerate(ordered, start=1):
        multiplier = total - rank + 1
        adjusted_running = max(adjusted_running, min(1.0, raw * multiplier))
        threshold = alpha / multiplier
        rejected_prefix = rejected_prefix and raw <= threshold
        output[name] = {
            "raw_p": raw,
            "holm_adjusted_p": adjusted_running,
            "holm_threshold": threshold,
            "reject": bool(rejected_prefix),
        }
    return output
