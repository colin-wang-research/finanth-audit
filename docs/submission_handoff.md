# KDD Submission Handoff

FinAuth-Audit is configured for KDD 2027 single-blind review. The repository
does not invent or infer author identities. A strict submission build is
therefore intentionally blocked until the submitting authors provide real ACM
metadata.

## Required Metadata

Edit `paper/author_metadata.tex`, or prepare a separate UTF-8 TeX file, with at
least one entry for each required field:

```tex
\author{First Author}
\affiliation{%
  \institution{Research Institution}
  \city{San Jose}
  \country{USA}
}
\email{first.author@example.org}
```

Multiple `\author`, `\affiliation`, and `\email` blocks are supported. ORCID,
author notes, and multiple affiliations may use the normal `acmart` syntax.

## Preflight

Validate the canonical file without compiling:

```bash
make submission-preflight
```

To validate a separate metadata file:

```bash
.venv/bin/python finauth_audit/paper/finalize_submission.py \
  --check-only \
  --author-metadata /secure/path/authors.tex
```

The preflight rejects placeholders, missing required fields, invalid email
syntax, anonymous metadata, and TeX commands that can read or write external
files.

## Finalize

External metadata may be preflighted without copying it into the paper tree,
but final compilation intentionally requires the canonical
`paper/author_metadata.tex`. This prevents a failed build from leaving a PDF or
HTML bundle whose displayed authors disagree with the restored source file.
After transferring the validated content into the canonical file, run:

```bash
make submission-finalize
```

The finalizer performs the following operations in order:

1. validates the metadata and KDD single-blind source mode;
2. compiles `paper/main.tex` rather than the anonymous internal wrapper;
3. rebuilds the page-faithful HTML from that PDF;
4. requires strict paper verification to pass every check;
5. writes the workspace-level `../paper/kdd/FinAuth-Audit-KDD.pdf` atomically;
6. writes a content-addressed submission manifest and PDF SHA-256 sidecar.

The manifest deliberately omits timestamps, hostnames, user names, and random
identifiers. It records only version, field counts, strict-verifier totals, and
hashes for the exact metadata and PDF supplied to the submission system.

The finalizer exits before compilation when placeholders remain. It never
modifies the frozen v0.6 scientific surface or reruns the one-time paper test.
The anonymous internal PDF uses the distinct filename
`../paper/kdd/FinAuth-Audit-KDD-INTERNAL-ANONYMOUS.pdf`; it must not be uploaded
as the single-blind submission.
