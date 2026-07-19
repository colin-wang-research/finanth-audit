# Training Utility Protocol Amendment, Round 71

## Timing and access boundary

This amendment was made during validation-only development. No controlled,
provenance, or public test outcome had been evaluated. The underpowered public
test remains sealed.

## Reason for amendment

Independent Fable review identified three protocol defects:

1. D0 used prediction-correctness labels while D1-D7 used authorization-safety
   labels, so D0 could not serve as a pure data-utility reference.
2. The validation mechanism `stress_only_period` was called compositional
   without generator evidence of cross-mechanism composition.
3. The protocol listed secondary learners that were not implemented in the
   registered validation smoke and used confirmatory wording despite
   `confirmatory: false`.

## Frozen correction

- D0 remains a prediction-only task-transfer diagnostic and is excluded from
  Holm-adjusted data-utility contrasts.
- D1 is the ordinary-only authorization reference.
- D2-D7 versus D1 form the six-member Holm family.
- `validation_unseen_controlled_compositional` is renamed
  `validation_unseen_controlled_stress_only_period`.
- Logistic regression is the only registered v0.2.0 learner. Secondary learner
  families require a future versioned protocol.

All corrected validation outputs must be regenerated. This amendment does not
authorize test access or a positive benchmark-training-utility claim.
