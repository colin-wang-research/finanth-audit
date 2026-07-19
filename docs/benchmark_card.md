# FinAuth-Audit Benchmark Card

## Purpose

FinAuth-Audit evaluates whether an execution rule is valid when a weak
financial prior is converted into an authorized action. It benchmarks rules,
not predictors and not trading profitability.

## Unit of evaluation

Each row is attached to an event cluster and contains a weak financial prior,
decision-time legal fields, source role, cost and risk fields, and outcome-only
labels. A prior may come from a heuristic, forecast model, policy, language
model, or evidence-only risk system. A rule emits `execute`, `reduce`, `review`,
or `abstain`.

## Released layers

- Controlled utility layer: 200,000 rows and 50,000 clusters.
- Controlled provenance layer: 200,000 rows and 50,000 clusters.
- Prospective sequential-market and institutional-workflow layers: 96,000 rows
  and 24,000 clusters.
- External order-book extension: 202,368 public Binance/MES-derived feature
  rows and aggregate result reports. Binance is a powered mixed-result arm;
  licensed MES is descriptive only and its row-level data are not redistributed.
- Actual-model proposal extension: 300 non-overlapping Binance UTC-date
  clusters across BTC, ETH, BNB, SOL, and XRP, with 50 development, 200
  one-time paper-test, and 50 author-unevaluated holdout dates. Three model
  configurations produce 1,200 paper-test proposals across directional and
  risk-limit tasks. Holdout proposals remain encrypted; rationales remain audit
  metadata and never become execution evidence.
- Public point-in-time event layer: frozen exploratory evidence; it does not
  pass the predeclared power gate and remains non-confirmatory.
- Training diagnostics: registered and prospective learner/curriculum runs
  with mechanism-wise coverage checks.

## Evaluation contract

The inspected paper test contains 5,000 controlled and 5,000 provenance
clusters. A separate 5,000-cluster community-hidden partition remains sealed.
The paper-test evaluator and scientific surface are content-addressed. Rules
cannot access realized utility, harm labels, laundering truth, tail loss,
future decoys, or test outcomes.

## Metrics

- Coverage: fraction of rows authorized for execute or reduce.
- Controlled NUAR: negative post-cost authorized outcomes divided by authorized
  rows; structural `N/A` when the denominator is zero. Frozen files retain the
  legacy field name `far` for compatibility.
- Actual-model consequence endpoints: directional economic-loss authorization,
  task-specific material harm, directional tail harm, and authority violation.
  Economic loss is not used as a synonym for illegality.
- ALR: authority laundering among authorized rows.
- CAU/MOC: cumulative authorized utility and missed opportunity cost.
- Review load: fraction routed to human review.
- Frozen policy profiles and conservative hypervolume: profile-specific and
  continuous summaries of joint NUAR/ALR/coverage constraints. The full
  certification volume remains a sensitivity analysis, not a universal score.

## Intended use

Use the artifact to compare authorization rules, audit field access, study
coverage collapse, test provenance integrity, and design future hidden-set
submissions. Do not use it as investment advice, a live trading system, an
approval system, or evidence of deployment readiness.

## Known limits

Controlled generators are author-implemented scenarios. Public event evidence
is underpowered for transfer claims. The external order-book extension uses
observed market data with constructed weak priors and assigned source roles;
the Binance intersection-union claim is intentionally retained as a mixed
failure. The actual-model layer observes proposals but still assigns legal
source roles and does not observe institutional permission. Its preregistered
inverse-rank hypothesis is not supported, and subgroup intervals are wide.
Review cost is
simulated rather than observed. Practitioner labels, independently authored
generators, and institutional deployment logs are not included.
