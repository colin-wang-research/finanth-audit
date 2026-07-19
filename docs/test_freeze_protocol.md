# Test Freeze Protocol

## Frozen before test access

- seed registry;
- split and event-cluster assignment;
- test stress mechanisms and severity levels;
- provenance attack definitions;
- legal feature lists by task;
- baseline implementations and field-access declarations;
- certification grid;
- D0-D7 training budgets and hyperparameter search spaces, with D0 registered as
  a prediction-only diagnostic and D2-D7 compared with D1 in the six-member
  data-utility Holm family;
- primary metrics and confidence-interval method;
- branch-neutral reporting template.
- branch-neutral generator parameters and mechanism holdout list;
- per-slice oracle-positive-opportunity definitions and expected rates;
- public-source independent-cluster and power gate;
- FRED confirmatory/exploratory vintage determination;
- traceable and untraceable laundering partitions;
- RQ4 learner classes, budgets, primary endpoint, and Holm family;
- AAAI overlap-audit configuration and result manifest.

## Certification grid

- harm thresholds: 0.05, 0.10, 0.15, 0.20, 0.25, 0.30;
- laundering thresholds: 0.00, 0.01, 0.02, 0.05;
- minimum coverage: 0.05, 0.10, 0.20, 0.30, 0.50.

A grid cell passes only when:

- UCB95(FAR) is no greater than the harm threshold;
- UCB95(ALR) is no greater than the laundering threshold;
- LCB95(coverage) is no less than the minimum coverage.

Certification volume is the passed-cell fraction. There is no arbitrary
weighted score.

The one-sided FAR, ALR, and coverage bounds are estimated once per rule and
evaluation slice. The 120 threshold cells are deterministic comparisons to
those bounds, not 120 independent hypothesis tests. Certification volume is a
descriptive operating-surface measure. Any across-rule inferential comparison
uses a separately declared primary endpoint and multiplicity correction.

## Test registry

Every test run records timestamp, code hash, config hash, data hash, command,
environment, output hashes, and whether results were inspected. Re-running a
test after inspection requires a documented defect, not a desired outcome.

## Prohibitions

- no test-driven generator edits;
- no regeneration to seek a preferred ranking;
- no hidden FinAuth-Worlds period access;
- no FinAuth-Worlds v5;
- no N/A-to-zero conversion;
- no baseline deletion because of weak performance;
- no row-level public-source significance;
- no result-dependent attack-definition changes.
