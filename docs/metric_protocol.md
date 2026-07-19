# Metric and Certification Protocol

## Decision categories

- `execute` and `reduce` are authorized actions;
- `review` is deferred to a separate process and is not execution coverage;
- `abstain` is no action;
- all four rates are reported separately.

## Denominators

- Coverage denominator: all rows in the declared evaluation profile.
- FAR denominator: authorized actions only. If zero actions are authorized, FAR
  is N/A, not zero.
- UPA denominator: authorized actions only. It is N/A at zero authorization.
- ALR denominator: authorized actions only. It is N/A at zero authorization.
- Review and abstain rates use all profile rows.
- No-opportunity controls are reported but excluded from certification and the
  coverage-collapse index.

## Controlled-layer uncertainty

All bounds use `event_cluster_id` as the resampling unit. Derived prior rows in
the same cluster are resampled together. Final certification uses 2,000
bootstrap replicates and seed 3794103132. One-sided 95% bounds are:

- 95th percentile for FAR and ALR UCBs;
- 5th percentile for coverage LCB;
- percentile intervals for descriptive secondary metrics.

Replicates with zero authorized actions contribute coverage zero but N/A FAR,
UPA, and ALR. If every replicate is N/A for an execution-denominator metric,
its bound is N/A and no certification cell depending on it passes.

## Coverage collapse

Stress coverage ratio is stress coverage divided by ordinary coverage. The
coverage-collapse index is one minus that ratio. It is N/A if ordinary
coverage is zero. Negative values are retained when stress coverage exceeds
ordinary coverage.

Per-slice oracle-positive-opportunity rates are disclosed before interpreting
collapse. Abstention in the explicit no-opportunity control is correct and does
not count as collapse.

Bootstrap and perturbation sub-seeds are derived from SHA-256 of the frozen
base seed plus a semantic label such as rule/profile or component-ablation.
No unexplained arithmetic offsets are used.

Coverage relations are reported explicitly: `UNIFORM` for stress/ordinary
ratio 1, `COLLAPSE` below 1, `EXPANSION` above 1, and
`UNDEFINED_ZERO_ORDINARY` when ordinary coverage is zero.

## Certification surface

The frozen grid is defined in `test_freeze_protocol.md`. Bounds are estimated
once per rule and profile. The threshold grid is a descriptive operating
surface, not 120 independent tests. We report overall and stress-profile
certification volumes and their minimum as the worst-profile volume.
