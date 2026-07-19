# Prospective Mechanistic Generator Robustness Protocol

## Purpose

The original utility-iid generator is not sufficient evidence of cross-DGP
validity. Round 75 prospectively adds two mechanistically distinct controlled
families before paper-test access:

1. `sequential_market`: persistent market state, regime jumps, tail events,
   inventory exposure, liquidity, volatility, and market-impact costs.
2. `institutional_workflow`: trading, credit, and research-to-action tasks with
   risk-budget utilization, compliance conflicts, approval-chain depth, and
   review-queue pressure.

These are author-implemented controlled generators, not external replications,
market simulators, institutional logs, or human approval labels.

## Frozen evaluation

- 12,000 independent event clusters per added family;
- four source priors per cluster;
- 60/20/20 train/validation/test split;
- validation split only in Round 75;
- unchanged v0.2 rule implementations and thresholds;
- 2,000 event-cluster bootstrap replicates;
- the same frozen certification surface;
- raw FAR and certification ranks compared across families;
- rule-by-generator ranges reported without selecting a global winner.

No generator parameter, rule threshold, or finding may be changed after the
FinAuth-Audit paper-test outcomes are inspected.

