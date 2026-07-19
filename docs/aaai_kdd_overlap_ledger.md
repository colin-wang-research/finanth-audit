# AAAI/KDD Overlap Ledger

## Firewall rule

Any P0 overlap in main claim, main data, main experiment, main figure, main
table, or headline number blocks paper writing until redesigned. Generic terms
such as coverage or false authorization may be shared only when their role in
the new benchmark is materially different.

| Axis | AAAI EPV exclusive surface | FinAuth-Audit v0.2.0 surface | Treatment | Residual risk |
|---|---|---|---|---|
| Main claim | A prior can earn execution authority through EPV | An authorization benchmark can falsely certify safety through validity failures | EPV absent from title, abstract, and contributions | Low if maintained |
| Problem | Prior-to-action validation method | Benchmark-validity audit | New RQs and certification object | Low |
| Method | Causal replay, LCB, split conformal, role-aware arbitration | No new authorization algorithm | Reuse only as equal-information baselines where legal | Medium |
| Data | AAAI controlled streams, finance rows, MiniWoB, coding/tool proxies | New controlled seeds/splits, coverage layer, provenance graphs, point-in-time event clusters, D0-D7 corpora | No AAAI split, seed, or result row reuse | Open until manifests exist |
| Coverage | Matched-coverage EPV validation | Coverage-collapse detection and minimum-coverage certification | Do not frame as EPV victory | Medium |
| Leakage | Clean-role source leakage | Role noise, delegation, paraphrase, semantic lineage, multi-hop laundering | Clean role retained only as easy control | Low after attacks exist |
| Public evidence | Bounded finance method evidence | Point-in-time public ranking transfer at event-cluster unit | No deployment language; no row-level p-values | High until timestamp audit passes |
| Training | Not a central contribution | D0-D7 benchmark training utility | Matched budget and hidden mechanism tests | Open |
| Metrics | FAR, coverage, utility, matched coverage | Certification volume, ALR, false block, safe delegation, collapse index, transfer regret | Shared primitives allowed; new validity metrics dominate | Medium |
| Figures | EPV gap/architecture, rank reversal, phase transition, clean-role leakage, matched coverage | Certified ranking, laundering surface, public transfer, training utility | Exact hash and semantic-layout audit required | Open |
| Tables | EPV result and ablation tables | Data/protocol summary, certification table, validity findings, artifact checklist | No numeric reuse | Open |
| Theory | Split-conformal theorem and role properties | No theorem reuse | Remove theorem/proposition framing from KDD | Low |
| Text | "Prediction is not permission" and EPV narrative | Validity-aware authorization benchmark | No paragraph reuse; new running example concerns benchmark gaming | Open until manuscript exists |
| Results | EPV harm reduction, matched coverage, MiniWoB, bounded finance | Raw-versus-certified ranking, laundering, controlled-public transfer, D0-D7 utility | Old numbers cannot be headline evidence | Low if enforced |

## Explicit exclusions

- No MiniWoB, coding, tool-call, or cross-domain results.
- No AAAI theorem or EPV algorithm section.
- No AAAI rank-reversal figure in the main paper.
- No clean-role leakage as the provenance contribution.
- No matched-coverage result presented as EPV superiority.
- No AAAI result number in an abstract, contribution, main figure, or main
  table.

## Required automated audits

1. Normalized 8-gram and 12-gram overlap across LaTeX and PDFs.
2. Exact and perceptual figure overlap.
3. Numeric token and source-table overlap.
4. Seed, split, and manifest overlap.
5. Contribution and claim-ledger comparison.

## Current status

The full mechanical audit is implemented and rerun against the current
manuscript and figures. The latest report records zero shared normalized 8-gram
and 12-gram passages, zero exact or perceptual figure matches, zero shared seed
tokens, zero exact data-file hashes, and zero forbidden headline occurrences.
See `results/overlap_audit/aaai_overlap_audit.md`. Claim-level review remains a
human responsibility, and the audit must be rerun after every manuscript or
figure change.
