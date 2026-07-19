# Powered Databento Replication Protocol v0.4

## Purpose and evidence status

This versioned extension creates a powered, temporally non-overlapping
replication of the frozen Databento arm from FinAuth-Audit v0.3. It uses MES
continuous-contract `bbo-1s` data from 2021-01-01 through 2025-07-31. The period
ends one day before the inspected v0.3 Databento period begins.

The direction of the v0.3 result is known. Accordingly, v0.4 is not described
as outcome-naive discovery. Its validity comes from temporal non-overlap,
unchanged protocol components, a pre-download written registration, a
pre-result power gate derived only from frozen v0.2 evidence, and exactly-once
test governance.

FinAuth-Audit remains a benchmark of execution rules across weak financial
priors. It is not a trading strategy, investment advice, a deployment claim, or
a profit leaderboard.

## Frozen clone from v0.3

The following components are copied unchanged from the frozen v0.3 Databento
arm:

- six weak-prior formulas and their source roles;
- seven registered execution rules and all thresholds;
- 09:30-16:00 America/New_York regular-session filtering;
- twelve decision times from 10:00 through 15:30 at a 30-minute stride;
- 30-minute strictly pre-decision feature windows;
- entry at the first valid BBO at least one second after decision;
- exit at the first valid BBO at least ten minutes after actual entry;
- five-second maximum quote age;
- exclusion of sessions containing more than one RTH `instrument_id`;
- exact one-contract bid/ask replay and a fixed 0.5 bps round-trip fee;
- session date as the independent cluster;
- co-primary endpoints, directions, SESOIs, and intersection-union decision;
- 10,000-replicate session-cluster bootstrap, seven-session moving-block
  sensitivity, and Holm-adjusted prior-family secondary tests.

Only the historical period and prespecified split sizes change: 120 structural
development sessions, 500 paper-test sessions, and all remaining valid sessions
as community hidden.

## Source and licensing

The source is licensed Databento `GLBX.MDP3` MES continuous `bbo-1s`. The
official preflight estimate is 87,534,222 records and USD 116.23. The user has
confirmed that this cost is not a binding limitation. Raw and row-level
Databento data are not redistributed; reproduction requires an appropriate
entitlement. Released material is limited to code, schemas, hashes, aggregate
results, and the written protocol.

## Temporal and label boundary

All deployable fields are observed or derived strictly before decision time.
Feature rows and sealed execution inputs are stored separately. The builder may
write entry and exit prices to sealed outcome files, but it may not compute
realized return, utility, harm, burden, or rule results. Labels are materialized
only after the content-addressed freeze is verified inside the write-once test
process.

Every source, decision, action, and outcome timestamp must remain within one
RTH session date, and every session must precede 2025-08-01. Sessions with
intraday continuous-contract instrument changes are excluded structurally.

## Co-primary inference

For each session and rule:

- false-authorization burden is false authorized rows divided by all prior
  rows;
- laundering burden is laundered authorized rows divided by all prior rows.

The preregistered comparison is Cost-Aware Gate minus Lifecycle Checklist. The
intersection-union result passes only when both conditions hold:

1. false-authorization burden has a point estimate at or below -0.015 and its
   two-sided 95% interval excludes zero in the negative direction; and
2. laundering burden has a point estimate at or above +0.020 and its two-sided
   95% interval excludes zero in the positive direction.

Power is recomputed before v0.4 outcomes using only the registered frozen v0.2
controlled paper-test cluster differences. At 500 test sessions, the expected
endpoint powers are 0.970 and 0.965, both above the 0.80 confirmatory gate.

## Governance

Before download, Fable and Supervisor review the written design. After data
construction, a structural audit verifies source hashes, date non-overlap,
timestamp ordering, quote staleness, session boundaries, instrument stability,
split disjointness, feature legality, and sealed-outcome alignment without
computing outcomes. The complete scientific surface is then archived by
content hash. The paper test may run exactly once. A failed execution is
retained and cannot be retried under v0.4. Community-hidden outcomes remain
unevaluated.

## Claim boundary

A pass supports temporal replication of the frozen burden trade-off on a
licensed MES BBO period. A failure, mixed result, or direction reversal is
retained and interpreted as evidence about mechanism dependence. No result is
deployment evidence, investment advice, or proof of trading profitability.
