# FinAuth-Audit Final Status Report

- Write time: 2026-07-19 20:20:00 CST
- Timezone: Asia/Shanghai
- Release version: **0.6.0**
- Submission state: **strict single-blind PDF and deterministic public package complete**

## 1. Canonical Deliverables

- Combined strict single-blind paper and supplement: `paper/FinAuth-Audit.pdf`
- Page-faithful HTML: `paper/html/FinAuth-Audit.html`
- Submission-workspace mirror: `../paper/kdd/FinAuth-Audit-KDD.pdf`
- Submission PDF SHA-256:
  `153834ba3fda929f34f322a17c1b14f1d76c3eb7b9e3454a02aac6ab04b61cb7`
- Public release policy: `release/release_spec.json`
- Public artifact verifier: `release/verify_release.py`
- Strict submission finalizer: `paper/finalize_submission.py`
- Author-metadata handoff: `docs/submission_handoff.md`
- v0.6 aggregate verifier report:
  `results/verification/public/real-agent-v06_verification.md`

The combined PDF has **20 pages**: eight main-content pages, one references
page, and eleven supplement pages. It contains five script-generated vector
figures, twenty script-generated tabular outputs, and 37 bibliography entries.
The first eight content pages comply with the KDD Datasets & Benchmarks page
budget; references and the supplement follow in the same PDF.

## 2. Paper And Visual Verification

- Strict single-blind paper verification: **111/111 pass**.
- Displayed authors: Ke Wang and Xiaorui Tang, with affiliations and emails.
- Undefined citations or references: **0**.
- Overfull horizontal boxes: **0**.
- Figure formats: PDF, SVG, and 360-dpi PNG.
- Minimum visible figure text: **8.0 pt**.
- Detected text, marker, legend, or boundary overlaps: **0**.
- All table environments use `\benchmarktableformat`; manual `resizebox`
  shrinking is prohibited.

The checked PDF is the strict single-blind build. `paper/main.tex` uses
`\documentclass[sigconf,review]{acmart}` and imports the completed canonical
`paper/author_metadata.tex`. The public PDF and submission-workspace mirror are
byte-identical at the SHA-256 above.

## 3. Benchmark Scope

FinAuth-Audit is a benchmark and release protocol for testing whether
comparisons among financial-agent authorization rules are valid under coverage,
provenance, consequence severity, review workload, point-in-time legality, and
evidence sufficiency. It benchmarks execution rules, not predictors or model
quality. It is not a trading strategy, investment-advice system, model
leaderboard, institutional approval study, or deployment validation.

The aggregate asset contains:

- **124,000** controlled, provenance, and mechanistic event clusters.
- **2,500** public event-market clusters, whose confirmatory public test remains
  sealed because the validation power gate is not met.
- **936** independent Binance dates and Databento MES sessions in the external
  order-book layers.
- A v0.6 actual-model partition of **300 independent UTC dates** across BTC,
  ETH, BNB, SOL, and XRP, with directional-execution and 25% risk-limit-increase
  tasks.

The actual-model partition is 50 development dates, 200 one-time paper-test
dates, and 50 author-unevaluated holdout dates. Three cached proposal sources
produce 1,500 plaintext development/paper-test rows. The paper test contains
1,200 rows over 200 independent date clusters. The 300 holdout proposal rows
were encrypted before outcome evaluation; the key remains outside the
repository and the ciphertext has not been decrypted.

## 4. Frozen v0.6 Result

The v0.6 surface was frozen before the one-time paper test with **727** pinned
surface hashes. The deterministic freeze archive SHA-256 is
`f26ff0d82645fbcc1692d95031c1a3c38420ce0ba37851d96ca1e550084885ff`.
The completed registry is read-only. The paper test must not be rerun.

The preregistered claim that controlled-to-actual-model directional
economic-loss rule ranks would remain inverse is **not supported**:

- Spearman rho: **+0.657** across six shared rules.
- Kendall tau-b: **+0.600**.
- Exact one-sided inverse-or-more-extreme permutation p-value: **0.932**.
- Date-cluster bootstrap median rho: **+0.543**.
- Bootstrap 95% interval: **[-0.657, +0.943]**.
- Bootstrap probability that rho is negative: **0.173**.
- Pairwise order reversals: **3/15**, all involving Lifecycle Checklist.
- Registered sensitivity omitting Lifecycle: rho **1.000**.

The earlier 60-date v0.5 inversion was a pilot that motivated this prospective
test. It is superseded for the inverse-transfer claim and is not a required
member of the v0.6 public release. No favorable subgroup replaces the registered
primary endpoint.

## 5. Consequence And Recalibration Findings

Economic loss, material harm, tail harm, and authority violation are separate
endpoints. On the v0.6 paper test, zero-shot economic-loss authorization rates
range from 0.641 to 0.694 among authorizing rules, material-harm rates range
from 0.138 to 0.238, and authority-violation rates range from zero to 0.088.
These values must not be interpreted as equivalent forms of harm.

Development-only recalibration is a secondary analysis that changes thresholds
but not rule forms. It restores coverage for several rules, but can worsen
authority violation and normalized utility. Lifecycle recalibration remains
structural `N/A` because none of the 36 development candidates met the frozen
0.10 coverage floor; the floor was not relaxed.

Raw proposal schema validity is 0.994. Nine expected records remain malformed
placeholders, there are zero repairs, and no outcome-conditioned or favorable
rerun is permitted. Rationales and self-reported evidence remain private audit
metadata and are prohibited from execution-rule evidence.

## 6. Other Binding Findings

1. **Raw NUAR can misrank authorization validity.** The frozen artifact field
   remains `far`, but it measures negative post-cost utility among authorized
   rows, not procedural illegality. Low NUAR can coexist with coverage collapse,
   lineage laundering, high review load, or missed opportunity.
2. **Current-role integrity does not establish lineage integrity.** A rule can
   eliminate direct role leakage while retaining indirect laundering through
   delegation or paraphrase.
3. **Missing lineage has an information cost.** Legal partial signatures reduce
   the empirical Bayes-risk lower bound but do not make hidden lineage fully
   identifiable.
4. **Mechanism stability is weaker than split stability.** Rankings that are
   stable across random splits can change across utility, sequential-market,
   and institutional-workflow generators.
5. **The powered external result is mixed and retained.** The Binance
   intersection-union claim fails because only one registered arm passes. MES is
   descriptive licensed evidence.
6. **Validity gates preserve negative and undefined results.** Underpowered
   public transfer and coverage-collapsed cross-mechanism training results remain
   exploratory or structural `N/A`.

No single execution rule is declared globally optimal. The benchmark reports
policy profiles, conservative hypervolume, Pareto trade-offs, and review costs
instead of a scalar trading-profit leaderboard.

## 7. Artifact And Release Safety

- FinAuth-Audit package tests: **216/216 pass**.
- Root integration tests: **57/57 pass**.
- Full v0.6 artifact verification: **771/771 pass**.
- Aggregate-only public verification: **8/8 pass**, with the full internal
  771/771 count recorded without private paths.
- Standalone public release tests: **6/6 pass**.
- Public paper verifier: **111/111 pass**.
- Public Python compilation: **pass**.
- Public release construction is allowlist-only and canonicalizes tar and gzip
  metadata for deterministic archives.
- Twelve v0.6 aggregate outputs are hash-pinned.
- Community-hidden, encrypted, sealed-outcome, row-level decision, provider
  response, rationale, credential, raw-cache, and licensed Databento files are
  excluded.
- The v0.5 pilot result files are excluded from the v0.6 public archive to
  prevent a superseded pilot from being mistaken for the binding result.
- Released Python files are compiled and the extracted package is installed and
  imported in a clean room.

The verified archive contains **222 members**. Its release verification report
and authoritative checksum sidecar remain external to the archive to avoid
self-reference. The authoritative archive SHA-256 is recorded only in the
external checksum sidecar and release metadata.

## 8. Review Status

The native Claude Fable Round 86 review bound the manuscript to the adverse
v0.6 result and required removal of the stale v0.5 headline. Round 87 confirmed
the claim-integrity boundary, prompted clearer malformed-record and inference
wording, and returned `READY_EXCEPT_REAL_AUTHOR_METADATA`. The final repository
review additionally detected and corrected stale v0.5 release metadata and
stale verification counts. Supervisor review uses the benchmark-paper,
pre-submission, and figure rubrics. Internal implementation-completion scores
are engineering gates, not guarantees of KDD acceptance or an award.

Fable Round 88 reviewed the metadata-to-submission workflow. The completed
finalizer rejects placeholders before LaTeX, requires the canonical metadata
source, compiles `main.tex`, rebuilds HTML, requires strict verification, and
writes a content-addressed PDF manifest and checksum. The public verifier uses
released aggregate v0.6 verification evidence rather than private freeze or
hidden-state manifests. Round 91 additionally checked the GitHub-ready package;
its stale-status finding is resolved by this report and by the byte-identical
public and canonical PDFs.

## 9. External-Only Limitations

The repository cannot supply or simulate:

- practitioner-elicited authorization thresholds;
- independently labeled institutional permission cases;
- observed human-review staffing and queue behavior;
- third-party custody of hidden proposal keys;
- a DOI-bearing archive record.

The actual-model layer is a bounded proposal-source study, not a complete
institutional-agent deployment. Public market outcomes are reconstructable, and
the historical-memory lexical audit cannot prove absence of implicit model
memory. These limitations remain explicit.

## 10. Submission Actions

1. Public source repository and version tag:
   `https://github.com/colin-wang-research/finanth-audit` at `v0.6.0`.
2. Versioned release assets are attached to the GitHub `v0.6.0` release:
   strict PDF, PDF checksum, verified archive, archive checksum, build manifest,
   and verification report.
3. Archive the GitHub release in Zenodo or another DOI-bearing repository and
   add the minted DOI without changing scientific results.
4. Submit the non-placeholder metadata and PDF to the KDD 2027 Datasets and
   Benchmarks Track Cycle 1 OpenReview entry.

Subject to those external publication actions, the v0.6 paper, supplement,
figures, tables, aggregate artifact, verifier, and public-release builder are
internally complete. This statement does not imply guaranteed KDD acceptance
or Best Paper selection.
