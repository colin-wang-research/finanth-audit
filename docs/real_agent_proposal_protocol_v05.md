# Real Agent Proposal Protocol v0.5

## Purpose

The v0.5 extension replaces deterministic agent stubs with cached outputs from
three real model configurations. Each model receives the same point-in-time
public market context and returns a structured weak financial prior. FinAuth-
Audit then evaluates execution rules over those proposals. The models are prior
sources, not execution authorities.

## Data And Independent Unit

The source is Binance USD-M Futures public `bookDepth` and one-minute kline
archives from 2025-01-01 through 2025-04-30. This period begins after the
inspected v0.3 Binance interval. One UTC date is one independent cluster. Each
date contributes one context at 12:00 UTC, with BTCUSDT and ETHUSDT assigned by
deterministic chronological alternation. The first 100 structurally valid dates
are frozen in chronological order: positions 1--20 are development, 21--80 are
paper test, and 81--100 are community hidden.

## Model Proposals

The registered configurations are GPT-5.4-mini, GPT-5.5, and Claude Sonnet 4.6.
They use temperature zero, the same prompt, and the same JSON schema. Raw output,
normalized output, model identifier, prompt hash, context hash, output hash, and
generation timestamp are cached. Requested model IDs, runner versions, and the
UTC generation window are recorded; provider snapshot IDs are recorded only
when the provider exposes them. Provider endpoints and credentials are not
stored or released. Rationale and action are produced in the same forward pass,
so rationale is not causally separable from proposal generation. It is retained
only for audit, excluded from every deployable rule, and is not inspected by a
human before the proposal cache is hashed and frozen.

The original transport registration used 40 contexts per request. Before any
outcome evaluation, the Claude CLI returned an incomplete fenced document after
seven context identifiers. Automated diagnostics inspected only wrapper shape,
byte count, token counts, and parse completeness. No proposal or rationale text
was reviewed. An independently reviewed pre-result amendment therefore changed
the common batch size to five for every model and quarantined every existing
cache, including complete GPT batches. The amendment was timestamped and
content-hashed before the first replacement call.

Transport normalization is deterministic and schema-agnostic. The loader first
attempts an exact JSON parse. Only when the entire response has an opening
Markdown JSON-fence line and a terminal fence line does it remove those two
wrapper lines and parse the unchanged interior. The normalization status is
logged for every response. It does not repair fields, values, missing records,
or invalid JSON.

A second pre-result transport amendment addressed the Claude CLI wrapper. A
complete five-context text response began with the non-semantic line, "I'll
analyze each context and provide trading proposals based on the pre-decision
fields:", before its fenced object. No proposal or rationale content was
inspected. The batch was quarantined. Replacement Claude calls use the official
JSON envelope and read only its top-level `structured_output` object. GPT calls
already use Codex `--output-schema`, which writes an exact JSON object. Thus the
model-facing prompt and common five-context grouping remain identical while
runner-specific envelopes are parsed explicitly.

The CLI did not populate `structured_output` while the schema contained the
top-level draft-2020 `$schema` declaration. Non-benchmark dummy probes isolated
that metadata declaration: removing only `$schema`, while leaving every field,
enum, range, and `maxItems` constraint unchanged, produced official
`structured_output` for both one and five dummy proposals. The final provider
schema therefore omits `$schema`. The previously considered `result` fallback
is disabled. Final outputs must be either an exact root JSON object from Codex
or a dict-valued official `structured_output` from Claude; every other envelope
shape is rejected. Failed Claude batches were quarantined and regenerated after
the final amendment was hash-locked.

The dummy probes contained no paper context, real action, sealed outcome, or
evaluation metric. Their outputs were never added to the proposal cache or test
registry and are excluded from every analysis. For retained GPT caches, the
manifest separately records the generation-time schema hash (with `$schema`)
and the final validation schema hash (without it); the declaration is metadata
only and does not alter application validation.

The reproducible Claude invocation is:

```text
claude --bare --print --model claude-sonnet-4-6 --output-format json \
  --permission-mode dontAsk --no-session-persistence --max-budget-usd 2.00 \
  --effort low --json-schema <registered-schema>
```

## Temporal And Outcome Boundary

Contexts contain only bars and depth observed at or before 12:00 UTC. Entry is
the 12:01 open and exit is the 12:31 open. Outcome prices and exit depth are
stored separately. Model generation occurs before the outcome evaluator is
authorized. Community-hidden contexts are not sent to any model in v0.5 and
their outcomes remain sealed.

## Evaluation

The rule thresholds are copied unchanged from the frozen FinAuth-Audit surface.
Model abstention remains abstention under every rule. Such contexts stay in the
task denominator with zero coverage contribution and are excluded from FAR's
numerator and denominator; coverage and FAR are always reported jointly. The paper test reports
coverage, FAR, CAU, MOC, review rate, overconfidence, cluster-bootstrap bounds,
and controlled-to-agent rule-rank correlation. This is a prospective descriptive
external validation, not a model leaderboard or a profitability study.

Depth snapshots may be up to 60 seconds old at decision time, and entry occurs
one minute later. This bounded staleness is reported as a limitation. The
exactly-once guarantee is procedural: a write-once registry records STARTED
before outcome materialization and is changed to COMPLETED or retained FAILED;
an existing registry prohibits rerun.

GPT configurations use the Codex Responses runner and Claude uses the Claude
Messages runner. Runner heterogeneity is recorded in the cache manifest and
schema-validity audit rather than treated as a model-quality difference.

## Claim Boundary

This layer establishes that the benchmark can ingest and audit actual cached
model proposals on a bounded public market task. It does not establish real
institutional permission, practitioner agreement, deployment readiness,
investment advice, or model superiority.
