# KDD Research Questions

## RQ1: Coverage Collapse

Can authorization rules appear safe by collapsing execution coverage,
deflecting difficult cases to review, or leaving stress periods entirely?

Required evidence:

- FAR at fixed coverage;
- coverage at fixed FAR;
- ordinary and stress coverage;
- execute, reduce, review, abstain, and N/A rates;
- missed opportunity and tail loss;
- coverage-collapse index;
- pre-frozen certification surface.

## RQ2: Provenance Laundering

Do rules that are safe under clean roles remain safe when role metadata is
missing or noisy and authority is delegated, paraphrased, or propagated across
multiple agent hops?

Required evidence:

- direct leakage and authority laundering rate;
- false block rate and safe delegation coverage;
- role-noise rates 0%, 5%, 10%, 20%, and 40%;
- hop depths 1 through 5;
- traceable and untraceable delegation;
- harm after laundering.
- separate traceable and untraceable ALR;
- detectable-in-principle laundering fraction.

## RQ3: Point-in-Time Transfer

Do controlled-benchmark rankings and metrics predict performance on public
event clusters whose fields are restricted to information available at the
decision timestamp?

Required evidence:

- controlled versus public Spearman and Kendall correlations;
- top-k overlap and winner change;
- FAR, coverage, calibration, and transfer gaps;
- ranking regret;
- event-cluster bootstrap confidence intervals;
- field-level availability and provenance audits.
- a pre-result independent-cluster and power gate; otherwise exploratory-only
  reporting.

## RQ4: Benchmark Training Utility

Does training on the full audit suite improve authorization generalization
relative to ordinary-only, stress-only, role-only, or cost-only authorization
training under matched model and hyperparameter budgets? A separate D0
diagnostic measures prediction-only-to-authorization task transfer and is not
interpreted as a pure data ablation.

Required evidence:

- D0-D7 matched training corpora;
- unseen severe and stress-only-period validation mechanisms, with genuinely
  compositional stress reserved for a separately defined sealed test;
- role noise, delegation, paraphrase, and multi-hop tests;
- public event-cluster transfer;
- FAR at fixed coverage and coverage at fixed FAR;
- authority laundering rate and certification volume;
- negative-transfer analysis when the full suite does not help.
- a pinned tabular learner claim, primary endpoint, matched budgets, and Holm
  correction.

## Supplemental sanity checks

Prediction/authorization rank reversal may be reported only as a supplemental
generator-sensitivity check. It is not a main RQ, main figure, contribution, or
headline claim.
