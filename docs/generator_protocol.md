# Controlled Generator Protocol

## Purpose

The controlled generator supports interpretable validity tests. It is not a
market simulator and cannot establish external financial validity by itself.
The generator must not be altered after test output to create a desired
coverage-collapse or laundering finding.

## Seed registry

Seeds are derived once from SHA-256 of the namespace
`finauth-audit-v0.2.0/<label>` and frozen before generation:

| Label | Seed |
|---|---:|
| smoke | 4134535595 |
| main-0 | 2683564103 |
| main-1 | 2012956673 |
| main-2 | 645167100 |
| main-3 | 1495259289 |
| main-4 | 1002717055 |
| bootstrap | 3794103132 |

No replacement seed is permitted because a result is weak or unfavorable.

## Branch-neutral construction

The generator samples latent opportunity, source reliability, costs, stress,
and provenance separately before computing outcomes. Rule outputs never enter
the generation process. No baseline-specific feature or threshold is used to
set labels.

### Latent opportunity

For each slice, the generator first samples a latent safe-opportunity state.
Except for the explicit no-opportunity control, each scored slice has a frozen
expected oracle-positive-opportunity rate between 0.15 and 0.55. The actual
realized rate is reported without post-hoc correction.

An oracle-positive opportunity means that at least one role- and
provenance-eligible full or reduced action has positive post-cost utility. This
label is outcome-only and never a deployable feature.

### Opportunity controls

- `ordinary`: expected opportunity 0.45;
- `moderate_stress`: 0.35;
- `severe_stress`: 0.20;
- `rare_safe_opportunity`: 0.15;
- `high_missed_opportunity`: 0.55 with higher safe-action value;
- `stress_only_period`: 0.20;
- `no_opportunity_control`: 0.00 and excluded from coverage-collapse scoring.

These are mixture probabilities, not guaranteed realized rates. The test report
must disclose the realized oracle-positive rate and confidence interval for
every slice.

## Mechanism holdouts

Training and validation may contain single-mechanism examples from:

- cost inflation;
- liquidity contraction;
- confidence corruption;
- role noise;
- single-hop delegation.

The confirmatory test includes mechanism-level holdouts:

- confidence-corruption plus liquidity contraction;
- cost underestimation plus high turnover;
- paraphrase laundering;
- multi-hop delegation depths 3 through 5;
- source-name and formatting shortcut reversals.

No held-out mechanism label or stress-family name is available to deployable
rules.

For provenance attacks, `detectable_in_principle` is true only when the full
lineage needed to identify laundering is attested. Delegation, paraphrase, and
multi-hop attacks are therefore detectable in principle only in the traceable
stratum; untraceable rows deliberately omit the lineage role chain and must be
reported separately rather than treated as undetected negatives.

## Negative controls

- no-opportunity slices where abstention is correct;
- clean provenance with legitimate delegation;
- role-neutral ordinary rows;
- permutation controls that break feature-label associations;
- duplicate-event controls grouped into one event cluster;
- masked stress/source identifiers.

## Component ablations

The evaluation must remove stress, provenance, cost, liquidity, role, and
public-calibration components one at a time. A controlled finding is not treated
as general if it disappears under small branch-neutral parameter changes or
depends on an exposed shortcut.

## Freeze and change policy

Generator code, distributions, mechanism holdouts, seeds, and expected
opportunity rates are hashed before test generation. After test inspection,
only correctness defects with a reproducible failing test may justify a new
version; the original result remains in the negative-results log.
