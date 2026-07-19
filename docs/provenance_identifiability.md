# Provenance Identifiability Boundary

FinAuth-Audit separates two questions.

1. **Traceable detection ceiling.** When an attested lineage chain is legally
   available, a hard provenance gate can reject any chain containing an
   ineligible role. ALR=0 in this stratum is a construction ceiling.
2. **Partial-observability trade-off.** When distinct latent lineage states share
   the same legal observable signature, no deterministic rule using that
   signature can distinguish every safe delegation from every laundering case.

The empirical ambiguity audit bins legal decision-time fields and computes the
minority-label mass within each observable signature. That mass is an empirical
Bayes-error lower bound for the chosen signature. It is conditional on the
controlled generator and is not a universal theorem about financial
institutions.

The appropriate benchmark question is therefore not whether a hard gate can
read a supplied lineage field. It is how ALR, safe-delegation coverage, false
blocking, review, and abstention trade off as lineage observability decreases.

