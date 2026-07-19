# FinAuth-Audit v0.6.0

Public artifact: <https://github.com/colin-wang-research/finanth-audit>

FinAuth-Audit is a validity-aware benchmark for financial-agent authorization.
It asks whether an authorization benchmark can make a rule look safe through
coverage collapse, provenance shortcuts, non-point-in-time public data, or
benchmark-specific training artifacts.

The controlled/provenance result key `far` is retained for release
compatibility, but the paper displays it as negative-utility authorization rate
(NUAR). It measures negative post-cost utility among authorized rows and is not
a procedural or legal violation label.

The project is isolated from:

- the AAAI EPV method paper;
- the frozen FinAuth-Bench v0.1.0 artifact;
- the closed FinAuth-Worlds Branch-N extension.

EPV is one baseline. It is not the benchmark identity, title, contribution, or
default recommended rule.

The repository contains a frozen controlled/provenance paper test, a separately
frozen external order-book test, and a prospective 300-date actual-model
extension. The latter uses five assets and two authorization tasks, selects
thresholds only on 50 development dates, evaluates 1,200 cached proposals once
on 200 paper-test dates, and preserves 50 author-unevaluated holdout dates as
encrypted proposal ciphertext. It is a rule-transfer task, not a model
leaderboard.

The v0.6 result does not support the preregistered inverse rank-transfer
hypothesis that was motivated by a 60-date pilot. It also reports economic
loss, material harm, tail harm, and authority violation separately, and retains
Lifecycle recalibration as structural `N/A` because no development candidate
met the frozen coverage floor.

The development/paper proposal cache has raw schema validity 0.994. Nine
expected records remain deterministic non-action placeholders in the task
denominators, with zero repairs and no favorable reruns.

Build the allowlist release with `make release`. The release
excludes community-hidden outcomes, raw provider envelopes, licensed Databento
row-level data, raw third-party archives, and credentials. It is a research
benchmark artifact, not investment advice, a live trading system, or deployment
approval.

## Quick Start

```bash
git clone https://github.com/colin-wang-research/finanth-audit.git
cd finauth-audit
python -m venv .venv
. .venv/bin/activate
python -m pip install ".[test]"
python -m pytest -q release_tests
```

The checked-in PDF, generated tables, vector figures, aggregate paper-test
outputs, manifests, and public verification reports are part of the release.

## Data Accessibility

All headline controlled/provenance benchmark functionality and the public
Binance order-book reproduction are available without redistributing licensed
MES records. The MES experiment is an optional licensed replication layer using
Databento `GLBX.MDP3` continuous MES `bbo-1s` data. Researchers with the
corresponding entitlement can reconstruct it from the public Databento dataset
entry at <https://databento.com/datasets/GLBX.MDP3> using the documented schema,
dates, session filters, scripts, manifests, and checksums. Row-level Databento
records are not redistributed because of license constraints.

## KDD Submission Finalization

Complete author metadata is required for the strict build. Validate it with
`make submission-preflight`, then run `make submission-finalize`. The finalizer compiles
`paper/main.tex`, requires strict paper verification, and writes the submission
PDF, manifest, and SHA-256 sidecar under the workspace-level `paper/kdd/`. The
anonymous internal mirror has the explicit suffix `INTERNAL-ANONYMOUS`. See
`docs/submission_handoff.md` for the ACM metadata format and external-file
workflow. The paper Makefile fixes `SOURCE_DATE_EPOCH` for v0.6.0, so repeated
builds from the same release produce the same PDF bytes and checksum.
