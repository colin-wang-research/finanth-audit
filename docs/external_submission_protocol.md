# External Submission Protocol

The community-hidden partition is reserved for evaluating new rules after the
paper release. It is not a second author-controlled test set.

## Submission contents

Submit:

- a rule implementation or container;
- a legal-field manifest;
- a rule classification and source-role declaration;
- deterministic per-row outputs with event-cluster identifiers;
- environment and dependency hashes;
- a short statement of intended use and limitations.

Allowed routes are `execute`, `reduce`, `review`, and `abstain`. Outcome fields,
future decoys, laundering truth, public outcomes, and hidden labels must not be
read by the rule.

## Maintainer evaluation

The maintainer runs the rule in an isolated environment against the sealed
community-hidden data. The evaluator records the command, code hash, field
access, output hash, and resource use. The submitter does not receive row-level
hidden labels.

## Reported results

The report includes coverage, NUAR (legacy result field `far`), ALR, CAU, MOC,
review load, safe-delegation coverage, cluster-aware uncertainty, and
certification profiles. NUAR and ALR are reported as structural `N/A` when no
rows are authorized. A single aggregate
score is not used as the acceptance criterion.

## Scientific boundary

Community-hidden results measure performance on this benchmark release. They do
not establish live trading profitability, institutional approval quality,
investment advice, or deployment readiness. Any extension using public or
institutional data must publish its own point-in-time, power, provenance, and
review-cost audit.
