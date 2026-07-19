# Negative Results Log

This file is append-only after experiment execution begins.

## Upstream evidence retained

- FinAuth-Worlds v4 reached Branch N and is closed. Shared learned rules did
  not recover a usable authorization region across the frozen worlds.
- The current FinAuth-Bench public replay shows complete abstention for its
  most conservative stress rule on all three public-source layers.
- The current rank-reversal finding changes materially between role-neutral
  and role-aware DGPs.
- The current FRED public extension is not vintage-aware.

These facts motivate FinAuth-Audit but are not FinAuth-Audit result claims.

## FinAuth-Audit results

### Development smoke 1

- Scope: validation-only smoke, 3,000 clusters and 12,000 rows.
- Result: all eight Phase-1 rules had certification volume zero under the
  frozen surface. Actual oracle-positive-opportunity rates were 0.55 to 0.91
  across scored slices, materially above the intended mixture probabilities.
- Interpretation: the non-opportunity mechanism still allowed small latent
  moves and outcome noise to exceed costs, so the opportunity-density control
  failed its own protocol. This is a generator-validity defect, not evidence
  that every rule should fail certification.
- Action: reduce non-opportunity latent moves and outcome noise using a
  rule-independent mechanism, then rerun validation smoke. Test data remain
  uninspected.

### Development smoke 2

- Opportunity-density control passed after the rule-independent repair: each
  scored slice's realized oracle-positive rate matched its sampled intent rate,
  and the no-opportunity control remained zero.
- All Phase-1 rules still had zero certification volume because legal prior
  features were too noisy to isolate opportunity; raw FAR remained above 0.55
  for most rules.
- Action: reduce source-signal noise and make decision-time confidence and
  expected edge reflect the already-sampled latent signal more faithfully. No
  rule output was used to generate outcomes, and test rows were not inspected.

### Development smoke 3

- Opportunity-density control continued to pass.
- `Lifecycle Checklist` achieved validation-only worst-profile certification
  volume 0.40 with coverage 0.337, FAR 0.157, and ALR 0.000. `No Action` had
  certification volume 0. `Cost-Aware Gate` had lower raw FAR (0.072) but
  failed certification because ALR was 0.243. This creates a development-only
  raw-versus-certified winner change without converting zero coverage to zero
  risk.
- Shortcut structure passed: no duplicate row IDs, no cluster split leakage,
  no identifier use, and the future decoy was present but unused. Source-name
  and stress-tag masking caused zero decision changes for every Phase-1 rule.
- These are validation smoke diagnostics, not test findings.

### Upstream drift detected by verifier

- The first strict smoke verification attempt failed because the three AAAI PDF
  build outputs changed after the initial upstream observation.
- FinAuth-Audit did not modify the AAAI repository. The AAAI `main.tex` and
  `supplement.tex` hashes remained unchanged, while figures/PDFs had been
  regenerated externally.
- The manifest now preserves both the initial and newly observed PDF hashes as
  record-only evidence. The overlap gate is bound to current AAAI source hashes
  and must be rerun when those sources change.

### Provenance laundering development smoke

- Scope: validation-only smoke, 3,000 clusters and 12,000 rows; 2,400 rows and
  600 clusters are evaluated. The confirmatory test split remains unopened.
- `Provenance Hard Gate` eliminates observed direct and indirect laundering on
  traceable authorized rows (ALR 0.000), but its overall FAR is 0.630. Provenance
  integrity therefore does not establish post-cost authorization quality.
- On untraceable rows, `Provenance Hard Gate` authorizes nothing, blocks all safe
  delegations, and has ALR reported as N/A rather than zero. This is a coverage
  collapse, not a safety victory.
- `Provenance Learned Gate` preserves 0.710 safe-delegation coverage on
  untraceable rows and has overall ALR 0.015, but this nonzero leakage is the
  cost of avoiding the hard gate's all-block policy. Neither rule dominates.
- `Hard Role Gate` removes direct role-ineligible execution but retains indirect
  leakage 0.171. The equal-information EPV adapter has the same indirect ALR,
  demonstrating that current-role checking does not audit upstream provenance.
- These findings are deterministic validation diagnostics under the controlled
  generator. They are not deployment evidence and will not be promoted to test
  claims until the complete benchmark, public point-in-time audit, and freeze
  protocol are satisfied.
- The validation smoke contains single-mechanism provenance attacks. Frozen
  compositional attacks are reserved for confirmatory test and will be reported
  separately; validation performance is not evidence of compositional
  generalization.

### Public point-in-time power gate, design iteration 1

- A new Polymarket point-in-time layer passed all structural checks with 3,000
  independent event clusters and 600 structurally assigned test clusters.
  Public test outcome metrics remained unopened.
- The first frozen common registry had 32 deployable configurations with at
  least 5% validation coverage. Under the predeclared cluster-uncertainty
  attenuation and Spearman simulation, estimated power was 0.7934, below the
  required 0.80. The source was therefore classified exploratory-only.
- Before test access, two pure fee-aware configurations were added because the
  registry lacked a baseline for Polymarket's documented probability-dependent
  taker fee. No FAR, CAU, public test outcome, or ranking was used to choose the
  fee thresholds; the initial failed gate remains retained here.

### Public point-in-time design iteration 2

- The first actual point-in-time fetch contained 3,000 clusters but covered only
  about 19 hours and was dominated by recurring short-horizon crypto markets and
  the fee-near-0.5 slice. It was rejected as externally narrow and archived as a
  development design rather than promoted.
- Monthly stratified metadata sampling from 2022-01 through a frozen
  2026-07-17 cutoff produced 3,800 candidate event clusters across 790 series.
  Twenty-three high-activity monthly windows reached the 20-page cap; this is a
  disclosed sampling limitation, not a claim of complete Polymarket history.
- CLOB price history returned sufficient sequences for only 516 candidates,
  almost entirely closing in 2026-06 or later. That failure was retained rather
  than backfilled with synthetic probabilities.
- A documented read-only Data API trades fallback recovered sufficient raw
  points for 2,712 additional candidates. The decision timestamp is frozen at
  35% of the public scheduled lifecycle; only the last pre-decision probability
  and first observed post-decision execution probability are used. Historical
  spread and depth remain unavailable and are not proxied.
- The decision timestamp is fixed at 35% of the public scheduled lifecycle.
  This replaced an initial 60% structural design after a no-outcome
  availability probe showed excessive early-close and missing-history loss at
  the higher fraction. No public test outcome or authorization metric was used
  for this revision.
- After fixed temporal and minimum-history exclusions, 2,626 event clusters
  were valid. The released development layer freezes an evenly spaced 2,500
  clusters with 1,500 train, 500 validation, and 500 structurally assigned test
  clusters. Test outcome metrics remain unopened.
- The validation-only cluster-power gate failed materially: estimated power was
  0.1635 and the one-sided 95% Monte Carlo lower bound was 0.1603 against the
  frozen 0.80 requirement. The Polymarket layer is therefore
  `exploratory_only`; it cannot support a confirmatory external-transfer claim.
- On validation, Direct Prior FAR was 0.198 and EPV Adapter FAR was 0.194, while
  EPV Adapter had lower CAU. This is a multi-objective trade-off and not an EPV
  victory. These validation diagnostics cannot be presented as public test
  findings.

### Training utility development iteration 1

- The first D0-D7 validation smoke averaged unseen controlled and provenance
  rows before checking per-mechanism coverage. It therefore appeared to give
  D7 FAR 0.000 at combined coverage 0.111.
- Domain decomposition showed that D7 controlled-unseen coverage was only
  0.0007. Role masking raised D7 combined FAR to 0.207 and ALR to 0.199, while
  its largest coefficients included confidence and risk-critic role indicators.
  The apparent broad improvement was therefore a coverage-collapse and role
  shortcut diagnostic, not a valid training-utility win.
- Before any test evaluation, the primary endpoint was repaired to follow the
  frozen protocol literally: FAR is computed separately for four frozen
  mechanisms and averaged only when every mechanism has defined FAR and a 95%
  coverage lower bound of at least 5%. The 5% floor is reused from the
  already-frozen certification grid. Invalid variants receive primary-endpoint
  N/A and cannot reject D0 through Holm correction.
- The original validation outputs are retained under
  `results/training_utility_smoke/design_iteration_1/`.

### Training utility protocol amendment and role-neutral control

- Strict verification exposed a real cross-layer holdout defect in the first
  D4/D5/D7 corpora: provenance-derived rows excluded provenance attacks but
  still inherited controlled test slices. The generator now applies both
  holdout families to every source layer before sampling. The affected corpora
  and all derived results were rebuilt; the invalid files are not used.
- Independent Fable review rejected D0 as a pure data-utility reference because
  D0 uses prediction-correctness labels while D1-D7 use authorization-safety
  labels. Before test access, D0 was frozen as a separate task-transfer
  diagnostic, D1 became the ordinary-only authorization reference, and D2-D7
  versus D1 became the six-member Holm family.
- The validation holdout previously called compositional is now named
  `stress_only_period`. No validation result is evidence of compositional
  generalization.
- In the corrected smoke, D1 has zero valid primary seed endpoints, so D2-D7
  have zero valid paired contrasts against D1. All Holm p-values are 1.0 and no
  hypothesis is rejected. This is retained as a failed data-utility validation,
  not tuned away.
- A matched D7 role-neutral training control masks all role and lineage fields
  during training, calibration, and evaluation. It has 0/5 valid primary seed
  endpoints; aggregate validation-unseen FAR is 0.062 at coverage 0.141, but
  both controlled holdouts fail the 5% coverage-LCB gate. It is excluded from
  the Holm family and cannot be presented as a success.
