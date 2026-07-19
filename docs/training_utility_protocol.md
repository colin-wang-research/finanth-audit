# Benchmark Training Utility Protocol

## Claim scope

RQ4 evaluates tabular authorization classifiers trained on FinAuth-Audit data.
It does not claim that an LLM or deployed financial agent improves by training
on the benchmark.

## Training datasets

The D0-D7 variants follow `data_protocol.md`. Each variant uses the same number
of training examples. When a component-specific dataset has fewer eligible
rows, deterministic cluster-aware sampling is used; no test row is used to
balance counts.

## Registered learner and endpoint

The primary learner in the v0.2.0 validation smoke is logistic regression with standardized numeric
features, one-hot categorical decision-time features, class weights fixed from
training data, `C` in `{0.1, 1.0, 10.0}`, and validation-only selection.

The primary endpoint is unseen-mechanism FAR at validation-fixed 20% coverage,
averaged first within event cluster and then across frozen validation-unseen
mechanisms and main seeds. D0 is a separate prediction-only task-transfer
diagnostic and is not part of the data-utility Holm family. D1 is the
ordinary-only authorization reference. D2-D7 are compared with D1 using Holm
correction over six predeclared contrasts.

A seed-level primary endpoint is defined only when every frozen mechanism has
a 95% lower confidence bound on coverage of at least 5% and a defined FAR. The
5% floor is the lowest value in the already-frozen certification grid. A
variant that collapses below this floor on any mechanism is reported as
primary-endpoint N/A and cannot win a Holm contrast through abstention.

No secondary learner is registered in the v0.2.0 validation smoke. Any future
learner-family robustness study requires a new versioned protocol and cannot
retroactively change this primary analysis.

## Role-neutral negative control

A D7 role-neutral control uses the same D7 training clusters, labels, logistic
model, seed list, C grid, calibration rule, and evaluation frames, but masks all
role and lineage fields consistently during training, calibration, and
evaluation. It diagnoses whether non-role evidence can support authorization
generalization. It is excluded from the Holm family and cannot establish a
positive training-utility claim.

## Secondary endpoints

- coverage at fixed FAR;
- authority laundering rate;
- certification volume;
- public transfer gap;
- missed opportunity;
- mechanism-specific negative transfer.

## Budget parity

All D0-D7 variants share architecture, training-example count, seed list,
optimization budget, hyperparameter grid, validation protocol, and stopping
rule within each learner family. Runtime and selected hyperparameters are
audited.

## Held-out tests

Future sealed test mechanisms may include compositional stress, paraphrase laundering,
multi-hop depths 3-5, source holdout, and public event clusters. Stress names,
test-family indicators, outcome fields, and hidden original roles are not legal
features.

The current validation-unseen controlled mechanism named
`stress_only_period` is a temporal holdout, not a compositional-stress claim.
