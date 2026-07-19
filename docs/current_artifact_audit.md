# Current Artifact Audit

## Scope

This audit establishes why FinAuth-Audit v0.2.0 must be a new benchmark rather
than another revision of FinAuth-Bench or the AAAI EPV paper.

## Upstream state

- FinAuth-Bench v0.1.0 is a complete controlled benchmark artifact with a
  200,000-row core, derived public-source extensions, multiple authorization
  rules, and a strict verifier.
- The current combined FinAuth-Bench PDF and release are frozen in
  `manifests/upstream_freeze.json`.
- FinAuth-Worlds is closed at Branch N. Its 2025-H1 hidden period remains
  unopened and is out of scope for FinAuth-Audit.
- The AAAI paper owns EPV, causal analogue replay, lower-confidence-bound and
  split-conformal authorization, execute/reduce/abstain method design,
  matched-coverage EPV validation, clean-role leakage, cross-domain proxies,
  MiniWoB, and bounded finance method evidence.

## Rejection-level validity risks in the current KDD artifact

### 1. Controlled-generator dependence

The current paper discloses that the role-neutral DGP preserves strong
prediction/authorization rank alignment while the main role-aware DGP creates
divergence. Therefore rank reversal is a generator-sensitive diagnostic, not a
stable benchmark-validity result. FinAuth-Audit removes it from the main RQs.

### 2. Coverage collapse

Several conservative rules obtain low FAR with very low or zero coverage on
public-source slices. A benchmark that reports FAR without coverage lower
bounds can reward all-abstain behavior. FinAuth-Audit makes minimum coverage a
certification constraint and reports N/A separately from zero.

### 3. Public-source pseudo-replication

The existing public layer derives 15,500 prior rows from 8,249 raw records:
400 Polymarket records become 1,500 rows, 1,222 GDELT records become 6,000
rows, and 6,627 FRED observations become 8,000 rows. Row-level uncertainty
would overstate independent evidence. FinAuth-Audit uses raw event clusters as
the inferential unit.

### 4. Point-in-time limitations

The existing FRED extension is explicitly not ALFRED-vintage-aware. It cannot
support a point-in-time macro-transfer claim. GDELT and Polymarket also require
field-level availability timestamps and provenance checks before they can
enter the confirmatory public-transfer task.

### 5. Clean-role leakage is too easy

If trusted `source_role` is directly available, a hard role gate achieves zero
direct leakage by construction. The new task must test missing or noisy roles,
delegation, paraphrase, semantic lineage, and multi-hop laundering.

### 6. Benchmark training utility is untested

The current artifact evaluates rules but does not show whether training on the
benchmark improves generalization to unseen stress, laundering, or public
event clusters. FinAuth-Audit preregisters matched D0-D7 training corpora.

### 7. Shortcut vulnerability is not a first-class audit

Identifiers, stress tags, source names, formatting, duplicate events, and
future-decoy fields can create artificial performance. FinAuth-Audit injects
and audits these shortcuts explicitly.

## Assets that may be reused

- schema and manifest infrastructure;
- baseline interface patterns;
- verifier and release-builder patterns;
- generic metric implementations after boundary review;
- selected ordinary controlled-row mechanisms with new seeds and splits;
- stress taxonomy names when they are not AAAI headline experiments.

## Assets that cannot be reused as KDD evidence

- AAAI text, theorem, figures, splits, seeds, and result numbers;
- MiniWoB, coding, tool-call, or cross-domain transfer results;
- EPV component or matched-coverage victory narratives;
- clean-role leakage as the main provenance contribution;
- FinAuth-Worlds hidden data or any v5 construct revision;
- row-level significance for public-source derived priors.

## Audit verdict

The new direction is justified only if the paper is organized around benchmark
validity and remains complete after removing EPV. Scope is accepted for
protocol review, not yet for experiment execution.
