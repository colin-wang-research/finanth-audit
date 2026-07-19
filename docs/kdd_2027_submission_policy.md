# KDD 2027 Datasets & Benchmarks Submission Policy

Source: https://kdd2027.kdd.org/datasets-and-benchmarks-track-call-for-papers/

Verified on 2026-07-18. The Datasets & Benchmarks review process is
single-blind: author names and affiliations must be listed. The official page
recommends `\documentclass[sigconf,review]{acmart}`. The submission is one PDF
with eight content pages, followed by references and an optional appendix with
no page limit.

`paper/main.tex` is the submission source and follows the single-blind class.
`paper/main_internal.tex` is an explicit anonymous wrapper used only for local
review builds while real author metadata is unavailable. Strict submission
verification must fail until `paper/author_metadata.tex` contains real names,
affiliations, cities, countries, and emails and the single-blind PDF has been
compiled from `main.tex`.
