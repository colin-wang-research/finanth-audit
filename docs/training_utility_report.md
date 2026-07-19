# Training Utility Report

## Scope

This is a validation-only tabular-learning diagnostic. It does not evaluate any controlled, provenance, or public test outcome and does not support an LLM-training or deployment claim.

## Matched design

D0-D7 each use 400 independent training clusters for each of 5 frozen seeds. Logistic regression is the primary learner; C and the authorization threshold are selected on seen validation mechanisms only.

## Primary finding

A primary endpoint is defined only when all four frozen mechanisms have a 95% coverage lower bound of at least 0.05. Valid seed counts are: {'D0': 5, 'D1': 0, 'D2': 0, 'D3': 0, 'D4': 0, 'D5': 1, 'D6': 0, 'D7': 0}. D0 is reported separately as a prediction-only task-transfer diagnostic. The registered data-utility family compares D2, D3, D4, D5, D6, D7 with D1; no contrast is valid in this smoke because the reference or candidate coverage collapses on at least one mechanism. FAR remains N/A where coverage collapses; it is not converted to zero.

## Shortcut and transfer diagnostics

D7 public-validation FAR is 0.113 at mean coverage 0.681. This is exploratory because the independent public power gate failed.
Masking role features raises D7 validation-unseen FAR to 0.167 and ALR to 0.165, showing material role-feature dependence.
The matched D7 role-neutral training control has 0/5 valid primary seed endpoints; its validation-unseen FAR is 0.062 at coverage 0.141. It is a negative control and is excluded from Holm contrasts.
On exploratory public validation, the role-neutral control has FAR 0.114 at coverage 0.695, compared with full D7 FAR 0.113 at coverage 0.681. This cross-domain asymmetry is descriptive; its cause is not identified by the current exploratory design, and no redundancy or causal explanation is claimed.

## Boundary

The smoke does not establish benchmark training utility. It establishes that the current fixed learner/data budget is vulnerable to mechanism-specific coverage collapse and role shortcuts. Negative transfer and invalid endpoints are retained.
