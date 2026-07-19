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

FinAuth-Audit shows that low negative-utility rates do not establish valid
authorization when coverage, lineage, transfer, and evidence sufficiency are
audited jointly.

## Abstract

Financial agents turn weak priors into actions, but outcome loss, policy
violation, and execution authority are distinct. FinAuth-Audit benchmarks
whether comparisons among execution rules remain valid when coverage,
provenance, consequence severity, review workload, utility, and point-in-time
evidence are audited jointly. We benchmark execution rules, not predictors.
The release contains 124,000 controlled, provenance, and mechanistic event
clusters, public event and order-book layers, and a preregistered 300-date
actual-model partition. Four findings define the benchmark. First, the lowest
negative-utility authorization rate (NUAR; the frozen artifact field is retained
as legacy FAR) is 0.064 but coexists with ALR 0.245, while lineage-clean rules
occupy different coverage, utility, and review frontiers. Second, a clean
current role retains 0.172 indirect laundering, and partial lineage signatures
leave a nonzero finite-sample information cost. Third, a 200-date actual-model
test does not reproduce a 60-date pilot inversion and shows that economic loss,
material harm, and authority violation rank operating points differently.
Fourth, predeclared validity gates retain a powered mixed Binance result, an
underpowered event-market layer, and coverage-collapsed learner comparisons
rather than promoting partial evidence. FinAuth-Audit is a release protocol,
not a trading strategy or deployment claim.

## Artifact

Public repository: <https://github.com/colin-wang-research/finanth-audit>

The public artifact excludes licensed Databento row-level records, raw provider
responses, credentials, and community-hidden material. The MES layer is an
optional licensed replication; the headline controlled/provenance benchmark
and public Binance reproduction do not require redistributed MES records.
