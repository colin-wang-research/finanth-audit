# Baseline Governance

## Registered Lifecycle Checklist timeline

- Round 68 plan registered the branch-neutral controlled benchmark and frozen
  certification task at 2026-07-17 19:22:17 CST, before Phase-1 data existed.
- `baselines/rules.py` and `configs/smoke.yaml` were created at
  2026-07-17 19:29:20 CST.
- The accepted validation Smoke 3 result was recorded at
  2026-07-17 19:45:32 CST.
- Two earlier generator designs failed and remain in the negative-results log.

This is not an external preregistration claim. The baseline was specified before
the accepted smoke result, but the generator was repaired during validation
development. The paper-test code, data, thresholds, and rule are therefore
frozen by content hash before one-time test access.

## Diagnostic component removals

- remove current-role eligibility;
- remove uncertainty/volatility review routing;
- remove execution-cost subtraction;
- remove the reduce route.

## Alternative compositions

- Role + Cost Gate;
- Role + Risk Gate;
- Role + Confidence + Cost Gate.

All variants use legal decision-time fields and receive no outcome, harm,
future-decoy, or test-label access. Their purpose is to test whether the
registered checklist's validation certification depends on one component or a
unique composition. They are not candidate methods and cannot be promoted after
paper-test inspection.

