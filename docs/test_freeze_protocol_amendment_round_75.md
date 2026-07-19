# Test Freeze Protocol Amendment, Round 75

## Timing and access boundary

This amendment is frozen before any FinAuth-Audit controlled or provenance test
outcome is inspected. The prohibited FinAuth-Worlds hidden period remains
untouched. Public test outcomes remain sealed.

## Validation-only robustness profiles

The following author-specified scenario profiles are frozen for threshold
sensitivity before paper-test access:

- frozen v0.2 grid;
- conservative institution;
- balanced platform;
- coverage-priority research;
- lineage-integrity-first.

Exact threshold arrays are code-frozen in
`evaluation/certification_robustness.py`. They are sensitivity scenarios, not
practitioner-elicited risk preferences and not test-tuned thresholds.

Continuous certification uses a fixed conservative reference box:

- FAR reference: 1.0;
- ALR reference: 0.25;
- coverage reference: 0.0.

The resulting hypervolume is descriptive. It does not include utility, trading
profit, reviewer accuracy, or a claim of global optimality.

## Review workload diagnostic

Review-capacity fractions are frozen at 5%, 10%, 20%, and 40% of evaluated
rows. Normalized review-cost scenarios are frozen at 0.5, 1, 2, and 5 basis
points per review. These are simulated workload sensitivities, not observed
human staffing costs or a human study.

## Baseline governance diagnostic

The registered Lifecycle Checklist is reproduced exactly, then evaluated with
four component removals and three independently motivated alternative
compositions. These variants are diagnostic baselines only. They cannot replace
the registered rule or be tuned after paper-test access.

## Prospective paper-test split

Before test outcomes are read, existing FinAuth-Audit controlled and provenance
test event clusters will be partitioned by a deterministic SHA-256 rule into:

- `paper_test`: one-time paper evaluation;
- `community_hidden`: remains sealed for future external submissions.

The partition uses event-cluster identifiers only. Outcome, harm, laundering,
utility, and opportunity fields are forbidden during partitioning. Exact cluster
lists, code/config/protocol hashes, and input data hashes must be frozen in a
content-addressed snapshot because this workspace has no Git commit history.

After paper-test inspection, no generator, rule, threshold, metric, attack,
split, or certification profile may be changed for v0.2. A rerun requires a
documented correctness defect and must preserve the original output.

