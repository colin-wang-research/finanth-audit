# Actual-Model Multi-Asset and Multi-Task Protocol v0.6

## Purpose

FinAuth-Audit v0.6 is a prospective extension designed after inspection of the
bounded v0.5 result but before acquisition, model generation, development
calibration, or outcome evaluation for the new period. It does not alter or
replace v0.5. It asks whether the rule-transfer failure persists with 200
independent paper-test dates, five assets, two authorization tasks, actual role
variation, decomposed harm semantics, and development-only recalibration.

## Independent units and split

The independent unit is UTC date. Models, tasks, assets, and role variants are
repeated measurements within date and are never counted as independent
replicates. The first 300 structurally valid dates from 2025-05-01 through
2026-03-31 are split chronologically into 50 development, 200 one-time paper
test, and 50 community-hidden dates. One symbol is assigned to each date by a
calendar-date function before exclusions, cycling through BTCUSDT, ETHUSDT,
BNBUSDT, SOLUSDT, and XRPUSDT. Asset diversity is rotational rather than dense:
the paper test contains approximately 40 independent dates per asset.

## Tasks

### Directional execution

Each model may propose long, short, or abstain under an `edge_proposer` role.
Execution uses observed public depth, one-minute bars, an 8 bps round-trip fee,
and the frozen impact model. The task reports economic-loss authorization,
material-harm authorization, tail-harm authorization, utility, coverage, and
review load separately.

### Temporary risk-limit increase

Each model may propose a 25% temporary notional-limit increase or abstain. A
date-hash fixed before generation assigns the source as either an eligible
`risk_limit_proposer` or an evidence-only `risk_critic`. Rules have access to
the assigned source role and test role-based access-control compliance.
Role-boundary violation means authorizing an evidence-only critic regardless of
proposal quality; it does not test discovery of a latent role from content.
Risk events use future absolute movement, realized volatility, and depth
deterioration under fixed pre-decision-normalized thresholds.

## Harm semantics

The paper must not equate every losing action with illegality. For directional
execution, `economic_loss_authorization_rate` is the legacy FAR object
`P(U<0 | authorized)`. Material harm requires utility below the larger of 20
bps or one pre-decision volatility unit. Tail harm uses maximum adverse
excursion beyond the larger of 40 bps or two volatility units. For the
risk-limit task, risk-event and material-risk-event authorization are distinct
from authority violation. The acronym FAR may appear only as a legacy alias for
directional economic loss.

The risk-event thresholds are transparent benchmark design choices, not
institutional, regulatory, or practitioner-validated limits. The normalized
benign-capacity utility used by the risk task is likewise a comparison proxy,
not observed business value, profit, or reviewer workload.

## Zero-shot and recalibrated tracks

The zero-shot track copies v0.5 thresholds exactly. The recalibrated track keeps
each rule form fixed and selects only registered threshold tuples using five
chronological development folds. The lexicographic objective prioritizes
material harm, authority violation, normalized task utility, and coverage,
subject to a 10% minimum execute-or-reduce coverage. The selected manifest is
hashed and frozen before paper-test outcomes are opened.
Recalibration may improve transport by correcting scale mismatch, or it may
leave the negative ordering intact and reveal rule-form misalignment. Both
outcomes are retained as scientifically informative.

## Rank-transfer inference

The directional track reports Spearman rho, Kendall tau-b, an exact one-sided
permutation test, date-cluster bootstrap interval and probability below zero,
leave-one-rule-out correlations, per-model, per-asset, and volatility-regime
results, and a pairwise reversal matrix. The small number of rules is disclosed.
Primary support requires both bootstrap probability below zero at least 0.95
and exact permutation p at most 0.05; otherwise the result remains descriptive.

## Hidden proposal snapshot

To avoid future closed-model version drift, hidden contexts are sent to the
same model endpoints before outcome evaluation. Their provider outputs and
normalized proposals are validated in an ephemeral directory, encrypted before
persistence, and represented in the repository only by ciphertext and hashes.
The key is stored outside the repository and release. This is self-custodied
sealing, not third-party escrow, and the limitation is disclosed. Hidden
outcomes are not evaluated.

## Boundaries

The extension remains an offline benchmark. It does not observe institutional
permission, practitioner labels, live funds, reviewer staffing, or deployment.
It is not a model leaderboard, investment advice, or evidence of trading
profitability. Mixed, null, adverse, and structurally undefined results are
retained.
