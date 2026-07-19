# OpenReview Submission Metadata

## Title

FinAuth-Audit: Validity-Aware Benchmarking of Financial Agent Authorization

## Authors

1. Ke Wang, first and corresponding author, Georgia Institute of Technology,
   `colinwang@gatech.edu`
2. Xiaorui Tang, Chanjin (Guangzhou) Technology Development Co., Ltd.,
   `alicetang0618@gmail.com`

Both authors must have complete OpenReview profiles before the author-registration
deadline. The submitted author order must remain consistent with the paper.

## Keywords

financial agents; execution authorization; benchmark validity; provenance
auditing; selective decision-making; abstention; agent safety

## TL;DR

FinAuth-Audit is a validity-aware benchmark showing that low false
authorization rates can conceal coverage collapse, authority laundering,
invalid transfer, and insufficient evidence in financial-agent execution
rules.

## Abstract

Financial agents increasingly turn weak financial priors into actions, yet a
low false authorization rate (FAR) can hide coverage collapse,
source-authority laundering, or insufficient evidence. We introduce
FinAuth-Audit, a benchmark and release protocol for testing whether comparisons
among execution rules remain valid under coverage, provenance, consequence
severity, review workload, utility, and point-in-time constraints. We benchmark
execution rules, not predictors. The artifact combines 124,000 controlled,
provenance, and mechanistic event clusters; 2,500 event-market clusters; 936
external market dates or sessions; and a prospective 300-date actual-model
partition. On a one-time controlled paper test, the lowest-FAR rule (0.064)
retains authority-laundering rate (ALR) 0.245, while lineage-clean alternatives
move along coverage, utility, and review frontiers. A clean current role still
retains 0.172 indirect laundering, consistent with a finite-sample Bayes-risk
limit under partial lineage observability. A separately preregistered test
evaluates 1,200 cached proposals on 200 dates. It does not reproduce a 60-date
pilot inversion: controlled-to-actual directional economic-loss ranks have
Spearman rho 0.657 and exact one-sided p-value 0.932, so the inverse hypothesis
is not supported. Economic loss, material harm, and authority violation also
produce different operating-point conclusions, while development-only
recalibration can restore coverage and increase authority violations. A powered
500-date Binance test passes only one of two co-primary arms, so its
intersection-union claim fails and is retained. FinAuth-Audit is not a model
leaderboard, trading-profit benchmark, investment-advice system, or deployment
claim.

## Artifact

Public repository: <https://github.com/colin-wang-research/finanth-audit>

The public artifact excludes licensed Databento row-level records, raw provider
responses, credentials, and community-hidden material. The MES layer is an
optional licensed replication; the headline controlled/provenance benchmark
and public Binance reproduction do not require redistributed MES records.
