# Public Replay Identifiability and Power Protocol

## Confirmatory sources

- Polymarket is confirmatory only when probability and any spread/order-book
  fields have historical timestamps no later than the decision time.
- GDELT is confirmatory only when event, ingestion, prior, action, and label
  times are present and ordered.
- FRED is exploratory in v0.2.0 unless ALFRED vintage observations and release
  timestamps are acquired and pass the temporal audit before result analysis.

The current non-vintage FRED extension cannot be promoted into the
point-in-time confirmatory layer.

## Independent unit

The inferential unit is `event_cluster_id`, not a derived prior row. Priors
derived from the same raw event remain in the same split and bootstrap cluster.

## Pre-result identifiability gate

RQ3 remains confirmatory only if all conditions hold before public test metrics
are inspected:

1. at least 200 independent test event clusters for each confirmatory source;
2. at least 500 independent test clusters combined;
3. no cluster appears in more than one split;
4. at least eight deployable rules have validation coverage of at least 5%;
5. all required point-in-time fields pass the availability audit;
6. a validation-only simulation estimates at least 80% power to distinguish
   absolute rank correlation 0.50 from 0 at two-sided alpha 0.05 under the
   cluster-bootstrap metric uncertainty.

If any condition fails, public transfer is reported as exploratory with full
intervals and no confirmatory winner claim.

## Statistics

- 2,000 event-cluster bootstrap replicates for final intervals;
- source-stratified resampling when sources are pooled;
- Spearman and Kendall rank correlation with intervals;
- top-k overlap, winner change, transfer gaps, and ranking regret;
- no row-level independent p-values;
- no selective reporting of only favorable transfer metrics.

## Multiple comparisons

The primary RQ3 endpoint is Spearman correlation of deployable-rule
certification volume between controlled validation and public test. FAR,
coverage, ALR, calibration, Kendall correlation, top-k overlap, and regret are
secondary descriptive endpoints. The primary endpoint is tested once; source-
specific and metric-specific results receive Holm correction within their
declared families.
