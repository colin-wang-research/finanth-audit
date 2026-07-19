# FinAuth-Audit

Validity-aware benchmarking of financial-agent authorization for the KDD 2027
Datasets and Benchmarks Track.

FinAuth-Audit evaluates execution rules, not predictors or trading models. Its
central question is whether an apparently low authorization-risk number remains
meaningful after auditing coverage, provenance, consequence severity, review
load, point-in-time legality, and evidence sufficiency.

## Reviewer Quick Path

1. Read the combined paper and supplement: [paper/FinAuth-Audit.pdf](paper/FinAuth-Audit.pdf).
2. Use the five-minute map: [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md).
3. Check scope and labels: [docs/benchmark_card.md](docs/benchmark_card.md) and
   [docs/metric_protocol.md](docs/metric_protocol.md).
4. Inspect the binding v0.6 aggregate result:
   [rank_transfer_zero_shot.json](results/real_agent_v06/paper_test/rank_transfer_zero_shot.json).
5. Inspect public verification:
   [real-agent-v06_verification.md](results/verification/public/real-agent-v06_verification.md).

The paper has 20 pages: main content on pages 1--8, references on page 9, and
the supplement on pages 10--20.

## Headline Findings

- The legacy artifact field `far` is displayed as negative-utility
  authorization rate (NUAR); it is not a procedural-illegality label.
- Low NUAR can coexist with coverage collapse, authority laundering, high
  review load, or missed opportunity.
- A clean current source role does not prove clean upstream lineage.
- The preregistered 200-date actual-model test does not reproduce the earlier
  60-date inverse-transfer pilot claim.
- Underpowered, mixed, or coverage-collapsed endpoints remain exploratory or
  structural `N/A` rather than being promoted as wins.

## Repository Map

| Path | Purpose |
|---|---|
| `paper/` | Combined PDF, LaTeX source, supplement, generated tables, and publication figures (Figure 1 is a reviewed high-resolution raster; Figures 2--5 are vector) |
| `evaluation/` | Authorization metrics, certification, provenance, transfer, and robustness code |
| `generators/` | Controlled, provenance, public, order-book, and mechanistic data builders |
| `results/` | Released aggregate paper-test and public verification outputs only |
| `docs/` | Benchmark, data, metric, release, and reproducibility protocols |
| `release/` | Allowlist builder and clean-room verifier |
| `release_tests/` | Public-package regression tests |

Internal logs, review transcripts, hidden proposal plaintext, hidden keys,
provider responses, row-level decisions, raw third-party archives, and licensed
Databento rows are not in this repository.

## Verify the Public Package

```bash
git clone https://github.com/colin-wang-research/finanth-audit.git
cd finanth-audit
python -m venv .venv
. .venv/bin/activate
python -m pip install ".[test]"
make public-check
make release
```

`make release` creates a deterministic allowlist archive and runs the public
clean-room verifier. The versioned GitHub Release also provides the strict PDF,
checksums, build manifest, archive, and verification report.

## Rebuild the Paper

```bash
make submission-preflight
make submission-finalize
```

The public package compiles the checked-in, hash-verified tables and figures.
Private frozen registries are deliberately excluded; the public finalizer does
not attempt to regenerate outputs that require those sealed inputs.

## Data Accessibility

Headline controlled/provenance functionality and the public Binance
order-book layer are reproducible without licensed market data. MES is an
optional replication layer using Databento `GLBX.MDP3` continuous MES `bbo-1s`
data. Entitled researchers can reconstruct it from
<https://databento.com/datasets/GLBX.MDP3>; row-level records are not
redistributed.

## Scope

This artifact is a benchmark and release protocol. It is not a trading
strategy, investment advice, a model leaderboard, institutional approval
evidence, or a deployment claim. See [ARTIFACT_CARD.md](ARTIFACT_CARD.md) for
the exact frozen-result and release boundary.
