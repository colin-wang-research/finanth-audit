# Data Protocol

## General principles

- Controlled rows are authorization scenarios, not market simulations or
  deployment logs.
- Public rows preserve raw event-cluster identity and field availability time.
- All decision features are point-in-time legal.
- Every data layer has a manifest, config hash, data hash, schema version,
  generation timestamp, seed registry, split registry, and provenance record.
- Test mechanisms and certification thresholds are frozen before test output is
  inspected.

## Layer A: Controlled Core

Use new seeds and new chronological or grouped splits. Ordinary scenarios and
controlled stress mechanisms support interpretable failures and a shared
baseline interface. No AAAI seed, split, table, or headline result is reused.

## Layer B: Coverage Audit

Include ordinary, moderate stress, severe stress, rare-safe-opportunity,
high-missed-opportunity, stress-only, and no-opportunity controls. Evaluate
target coverages 5%, 10%, 20%, 40%, 60%, and 80%. Preserve the four-way route:
execute, reduce, review, and abstain.

## Layer C: Provenance Laundering

Each record contains:

- row and event-cluster IDs;
- original and current source IDs;
- claimed and verified roles;
- delegation parent;
- transformation chain and evidence lineage;
- hop depth and content hash;
- prior action, confidence, uncertainty, costs, liquidity;
- decision and outcome timestamps;
- direct, indirect, and authority-laundering labels.

Attacks are frozen as role noise, delegation, paraphrase laundering, and
multi-hop laundering. Attack definitions cannot be changed after test output.
Traceable and untraceable laundering are separate evaluation strata. The
traceable stratum exposes legal lineage evidence from which detection is
possible; the untraceable stratum is an impossibility/stress control and cannot
be pooled into a headline ALR without separate reporting.

## Layer D: Point-in-Time Public Replay

The statistical unit is the raw event cluster.

### FRED

Use ALFRED vintages when possible. If vintages are unavailable, label the layer
exploratory and do not call it point-in-time FRED. A revised macro value is
never a decision feature before its vintage release.
For v0.2.0, the current non-vintage FRED cache is frozen as exploratory unless
a new ALFRED-vintage layer passes the availability audit before public result
inspection.

### GDELT

Record event time, ingestion time, prior time, action time, and label time.
Derived priors remain grouped by the raw event cluster.

### Polymarket

Retain only historically observed fields. Probability, timestamp, spread,
resolution rule, resolution time, and dispute status require provenance.
Depth is omitted unless historical order-book snapshots exist. Simulated depth
cannot be described as observed.

### Inference

All confidence intervals and tests use event-cluster bootstrap or an equivalent
cluster-aware procedure. Row-level independent p-values are prohibited.

## Layer E: Training Utility

Freeze matched datasets:

- D0 prediction-only task-transfer diagnostic, excluded from data-utility Holm contrasts;
- D1 ordinary-only authorization reference;
- D2 Stress only;
- D3 Ordinary plus stress;
- D4 Ordinary plus stress plus provenance;
- D5 Role only;
- D6 Cost only;
- D7 Full audit suite.

Models, sample budgets, hyperparameter budgets, seeds, and stopping rules are
matched. D2-D7 are compared with D1 for data utility; D0 is reported separately
because its supervision target is prediction correctness rather than
authorization safety. Negative transfer is retained and analyzed.

## Layer F: Actual-Model Proposals

Use a source period that does not overlap any inspected order-book result.
Every model configuration receives the same decision-time context fields and
must emit the registered structured schema. Cache raw transports locally and a
normalized proposal table with prompt, context, output, runner, generation
schema, and validation schema hashes. Report raw schema validity, repairs,
model abstention, and source-quality diagnostics before rule outcomes.

Model rationales, self-reported evidence, and tool narratives are audit
metadata only. They are prohibited from execution rules. The benchmark assigns
the same proposer eligibility to every configuration, so the task evaluates
rule transfer across prior sources rather than institutional permission or
model superiority. Development and paper-test contexts may be generated;
community-hidden contexts remain unsent and their outcomes unevaluated.
