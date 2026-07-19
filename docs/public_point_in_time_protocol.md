# Public Point-in-Time Protocol

## Scope

The Polymarket layer is an offline public-source transfer diagnostic. It is not
deployment evidence, a profitability study, or a claim that Polymarket events
represent the full financial-agent population. The inferential unit is one raw
event cluster, and the released development layer contains one row per cluster.

## Frozen sampling design

- Metadata window: 2022-01-01 through 2026-07-17T13:05:00Z.
- Eligibility: resolved binary CLOB market, at least 24 hours of scheduled
  duration, at least USD 1 recorded volume, and a public scheduled end time.
- Monthly cap: 120 event clusters.
- Series cap: 12 clusters per series within a month.
- Candidate cap: 3,800 clusters, evenly spaced over candidate close time after
  monthly and series controls.
- Final development layer: 2,500 structurally valid clusters, split
  chronologically into 1,500 train, 500 validation, and 500 sealed test
  clusters.

The Gamma offset API required 585 pages. Twenty-three high-activity monthly
windows reached the 20-page limit. The artifact therefore represents a frozen,
stratified public sample rather than an exhaustive historical census.

## Historical probability provenance

The preferred source is the public CLOB batch price-history endpoint. Because
that endpoint returned sufficient history for only 516 candidates, the builder
uses the documented public Data API trades endpoint as a fallback. YES-token
trades map directly to YES probability; NO-token trades map to one minus the
trade price. User identity, profile text, and natural-language rationale are not
used by any execution rule.

Every row records `historical_probability_source` as either
`clob_prices_history` or `data_api_trade`. Historical spread and depth are not
available and are omitted rather than imputed.

## Decision-time rule

The decision timestamp is fixed before outcome evaluation:

```text
decision_time = created_time + 0.35 * (scheduled_end_time - created_time)
```

The source probability is the last observed probability strictly before the
decision timestamp. The execution probability is the first observed
probability strictly after the decision timestamp and before resolution. A row
is excluded if it lacks four pre-decision observations, a post-decision
execution observation, a valid public lifecycle, or an observed fee schedule
when fees are enabled.

Resolution outcome and post-decision utility are labels only. They are not
legal baseline features. The point-in-time audit mechanically verifies
`source < decision < action < outcome`, the fixed lifecycle formula, one row per
event cluster, split isolation, and source provenance.

## Inferential boundary

The public test contains 500 clusters but remains sealed because the
validation-only power gate estimated power at 0.1635, below the frozen 0.80
requirement. The source is classified `exploratory_only`. Public validation
diagnostics can motivate future data collection but cannot be reported as
confirmatory external-transfer results.
