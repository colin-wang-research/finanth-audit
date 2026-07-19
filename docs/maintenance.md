# Maintenance and Versioning

## Frozen releases

An inspected paper-test release is immutable. Its generators, rules,
thresholds, metrics, attacks, splits, certification profiles, and primary
endpoints are bound by hashes. A correction must preserve the original output,
identify the defect, and create a new versioned archive.

## New benchmark layers

New data layers must provide:

1. a schema and legal-field manifest;
2. event-cluster identifiers and point-in-time alignment;
3. provenance and raw/processed hashes;
4. a documented label boundary;
5. train/validation/test or hidden-submission separation;
6. focused tests and a reproducible build command;
7. a limitation statement explaining external validity.

Adding rows to an existing inspected layer without a new version is not
allowed. Derived rows must not be counted as independent event clusters.

## New execution rules

Contributors declare their rule class, legal fields, source-role assumptions,
and whether the rule is deployable, diagnostic, or oracle. The feature-access
audit rejects outcome fields and future decoys. Results must preserve `N/A` for
zero authorization and must report coverage, review load, and provenance
metrics together with FAR.

## Regression gates

Every release runs unit tests, figure QA, paper verification, artifact hash
verification, and a negative-field access audit. Public or institutional
extensions require a separate power analysis and must not be described as
confirmatory until the registered gate passes.

## Governance

Issue reports should identify a version, command, input hashes, and the smallest
reproduction. Maintainers should preserve negative results and deprecated
versions rather than silently rewriting the evaluation history.
