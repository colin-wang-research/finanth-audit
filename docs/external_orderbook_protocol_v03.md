# External Order-Book Protocol v0.3

## Purpose

This versioned extension addresses the largest remaining validity limitation in
FinAuth-Audit: the absence of a powered public layer with observed execution
frictions. It does not modify the frozen v0.2 controlled or provenance paper
test. The extension asks whether a preregistered authorization trade-off is
observable in public or licensed point-in-time order-book data.

FinAuth-Audit remains a benchmark of execution rules across weak financial
priors. It is not a trading strategy, an investment-advice system, or a profit
leaderboard.

## Evidence tiers

1. **Confirmatory public replication:** Binance USD-M futures public
   `bookDepth` snapshots joined to public 1-minute klines. The independent unit
   is UTC date, shared across both symbols, all decisions, and all prior
   families.
2. **Descriptive licensed replication:** Databento `GLBX.MDP3` MES continuous
   `bbo-1s`. The independent unit is the RTH session date. The frozen 140-session
   test is underpowered for the co-primary claim and therefore remains
   descriptive.
3. **Community hidden:** all remaining structurally valid dates or sessions are
   kept unevaluated for future external submissions.

## Binance construction

- Sources: official daily USD-M futures `bookDepth` ZIP files and official
  monthly 1-minute kline ZIP files for BTCUSDT and ETHUSDT, 2023-01-01 through
  2024-12-31.
- Source integrity: every archive must have an official checksum and a matching
  local SHA-256 digest.
- Decisions: 01:00 through 22:00 UTC at a 60-minute stride.
- Features: only closed bars ending before the decision and the latest
  non-stale depth snapshot at or before the decision.
- Entry: next 1-minute open. Exit: the open 30 minutes later. The next decision
  occurs 29 minutes after the outcome is observed.
- Independent cluster: UTC date, not symbol-date or prior row.
- Public depth snapshots arrive approximately every 30 seconds. The frozen
  maximum age is 60 seconds, allowing at most one missed update while excluding
  older snapshots.

### Cost model

The fixed action notional is USD 100,000. At each side of the trade, the depth
approximation is

```text
impact_bps = min(100, 50 * action_notional_usd / side_depth_at_1pct)
```

The deployable cost field uses twice the current-side impact estimate and the
frozen round-trip fee. The label uses the decision-side entry impact and the
outcome-side exit impact. The primary round-trip fee is 8 bps; 4 and 12 bps are
frozen sensitivity conditions. The approximation is intentionally simple and
is not presented as a full market-impact model.

Feature rows and sealed execution inputs are stored separately. Builders write
entry/exit prices and outcome-side depth to `*_outcomes.parquet`, but do not
compute realized return, utility, harm labels, or rule results. The one-time
test process materializes those labels only after it verifies the
content-addressed freeze.

## Databento construction

- Source: the locally cached, hash-verified `GLBX.MDP3` MES continuous
  `bbo-1s` files listed in the upstream Databento manifest.
- Session: 09:30-16:00 America/New_York.
- Decisions: 10:00 through 15:30 at a 30-minute stride.
- Entry: first valid quote at least one second after decision. Exit: first valid
  quote at least ten minutes after entry.
- Sessions with more than one `instrument_id` in the evaluated RTH window are
  excluded before outcomes are inspected.
- Long labels buy at ask and sell at bid; short labels sell at bid and buy at
  ask. A fixed 0.5 bps round-trip benchmark fee is applied.
- Raw or row-level Databento data are not redistributed. Reproduction requires
  the corresponding entitlement.

## Weak-prior families

The six frozen prior families are momentum, reversal, displayed-depth
imbalance, volatility breakout, a deliberately spread-blind high-confidence
proposer, and a role-ineligible liquidity critic. They are repeated
measurements within each independent date cluster, not independent samples.
The same v0.2 execution-rule registry and thresholds are used without external
outcome tuning.

## Primary inference

For each date and rule:

- false-authorization burden is false authorized rows divided by all prior rows;
- laundering burden is laundered authorized rows divided by all prior rows.

The preregistered comparison is Cost-Aware Gate minus Lifecycle Checklist. The
confirmatory result passes only if both conditions hold:

1. the two-sided 95% interval for false-authorization burden excludes zero in
   the negative direction and the point estimate is at most -0.015; and
2. the lower side of the two-sided 95% interval for laundering burden is above
   zero and the point estimate is at least +0.020.

Inference resamples dates with 10,000 replicates. A seven-day moving-block
bootstrap is a dependence sensitivity. Prior-family secondary tests use Holm
correction.

The seven-day block is a preregistered calendar-cycle sensitivity rather than
an autocorrelation-tuned block length. It tests whether the conclusion survives
short weekly dependence without presenting the block choice as optimal.

## Power gate

Power is computed before external outcomes from the frozen v0.2 controlled
paper-test cluster differences. With 500 Binance dates, the two expected powers
are 0.970 and 0.965. With 140 Databento sessions, they are 0.528 and 0.514, so
Databento cannot support the confirmatory claim.

## Freeze and one-time execution

The source manifests, split membership, configuration, preregistration, feature
access manifest, generator code, evaluator code, tests, and pre-result power
report are content-addressed before any external test metric is computed. The
test registry is write-once. Any rerun or hidden-set access is an error.
If execution crashes after the registry is created, the failed attempt remains
registered and cannot be silently retried; a new versioned preregistration is
required. Fee-grid sensitivities use 1,000 bootstrap replicates, while both
co-primary endpoints use the full 10,000.

## Limitations

- Binance `bookDepth` provides banded depth rather than message-level order-book
  reconstruction.
- The linear impact approximation is a benchmark stress model, not a calibrated
  execution simulator.
- Databento is a licensed replication with fewer independent sessions and no
  redistributable row-level release.
- Neither source provides practitioner authorization labels or observed human
  review operations.
