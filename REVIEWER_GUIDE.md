# Reviewer Guide

## Five-Minute Review

1. **Problem and contribution:** paper pages 1--2.
2. **Task, labels, and partial observability:** pages 2--4.
3. **Core controlled and provenance findings:** pages 5--7.
4. **Prospective actual-model and validity-gate findings:** pages 7--8.
5. **Reproduction and full evidence:** supplement pages 10--20.

## Four Claims to Check

| Threat | Naive conclusion | FinAuth-Audit check |
|---|---|---|
| Coverage collapse | Lowest NUAR is safest | Joint NUAR, ALR, coverage, utility, and review reporting |
| Provenance laundering | Current role is sufficient | Direct and indirect lineage audit under partial observability |
| Transfer failure | Controlled ranking transfers | Frozen prospective actual-model test |
| Evidence insufficiency | Partial evidence supports a claim | Power, mechanism-coverage, and structural-N/A gates |

## Binding Evidence

- `results/paper_test/report.md`
- `results/real_agent_v06/paper_test/rank_transfer_zero_shot.json`
- `results/real_agent_v06/paper_test/zero_shot/metrics.csv`
- `results/real_agent_v06/paper_test/recalibrated/metrics.csv`
- `results/external_orderbook_v03/paper_test/binance/primary_endpoints.json`
- `results/verification/public/real-agent-v06_verification.md`

## Reproduction Commands

```bash
python -m pip install ".[test]"
make public-check
make submission-finalize
make release
```

Expected public gates for v0.6.0:

- strict paper verification: 111/111;
- public release tests: 7/7;
- aggregate v0.6 verification: 771/771 internally, summarized as 8/8;
- clean-room release verification: pass.

## Deliberate Exclusions

The public package excludes hidden proposal plaintext and keys, provider
responses and rationales, row-level decisions, raw caches, licensed Databento
rows, internal logs, and review transcripts. These exclusions are benchmark
governance boundaries, not missing reviewer evidence.
