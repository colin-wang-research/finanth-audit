# Public Release Policy

FinAuth-Audit publishes an allowlist-built v0.6 review artifact rather than copying
the working directory. The release builder includes code, documentation, the
combined paper, generated figures and tables, opened controlled/provenance
paper-test rows, validation-only mechanism rows, public Binance feature rows,
and aggregate result files.

The builder rejects any member path containing a sealed or community-hidden
surface, any `*_outcomes` file, licensed Databento row-level data, credentials,
cache directories, or result-level decision dumps. The release intentionally
omits raw third-party archives and hidden challenge identifiers/outcomes.

The v0.6 actual-model namespace is closed by default. It permits only the exact
hash-pinned aggregate metrics, bootstrap bounds, subgroup diagnostics, rank
inference, source-quality summaries, historical-memory aggregates, and public
verification reports named in `release/release_spec.json`. It rejects every
proposal, context, decision, rationale, raw provider envelope, free-text result
summary, hidden ciphertext, and key, even if an unexpected file uses otherwise
innocuous column names.

The earlier v0.5 actual-model aggregates are pilot evidence superseded for the
registered inverse-transfer claim. They remain in the private audit repository
but are not members of the v0.6 public release. The verifier rejects any
`results/real_agent_v05/` member, requires `FINAL_STATUS_REPORT.md` to identify
version 0.6.0 and the binding `rho=+0.657`, `p=0.932` non-replication, and rejects
a status report that promotes the old negative pilot correlation.

Build and verify the archive with:

```bash
make -C finauth_audit release
```

The command regenerates the paper, runs internal paper verification and the
frozen legacy and v0.6 verifiers, creates a byte-canonical deterministic
`tar.gz`, extracts
it into a temporary clean room, verifies every SHA-256 entry against
`RELEASE_MANIFEST.json`, compiles all packaged Python files, scans packaged CSV
split columns, and fails if a hidden or licensed row-level surface is present.
The verifier also checks pinned v0.6 member hashes, canonical member order, tar
metadata, gzip timestamp, and byte-for-byte archive reconstruction.
It also checks release-level claim consistency so that path and hash integrity
cannot mask stale scientific conclusions.

The public artifact includes `paper/finalize_submission.py` and
`docs/submission_handoff.md`. The finalizer accepts only the canonical author
metadata file for compilation, fails before LaTeX while placeholders remain,
and requires the strict paper verifier before publishing a submission PDF.

The release archive is a reproducible research artifact. It is not investment
advice, a deployment approval, or permission to redistribute upstream data.
