# Commutator Ladder Rotation Nested V1 Findings - 2026-04-18

## Setup

- Benchmark family: `matched_rotate_alpha`
- Split: `cartesian_blocks`
- Hyperparameter protocol: nested inner validation on a structured inner split
- Selection metric: inner validation reconstruction MSE
- Variants:
  - `baseline`
  - `smooth`
  - fixed `cfp_l0.010`
  - fixed `cfp_l0.050`
  - `step_selected`
  - `cfp_selected`

## Main Result

Rotation remains mixed, and nested selection does not rescue it into a clean CFP story.

## What Happened

- At `matched_rotate_5.00`, fixed CFP is still the best result:
  - `cfp_l0.050` `0.000849`
  - baseline `0.001071`
  - `cfp_selected` only `0.000962`
- At `matched_rotate_10.00` and `20.00`, baseline stays best.
- At `matched_rotate_30.00`, `step_selected` is best: `0.002771` vs baseline `0.002992`, while `cfp_selected` is effectively tied with baseline.

## Interpretation

This family now points to two risks:

1. the current nested selector is unstable for CFP, because it fails to recover the manually good low-angle setting;
2. some of the late-ladder gain is better explained by a step prior than by CFP itself.

## Research Decision

- Keep `rotation` as a secondary diagnostic family, not as the headline support.
- Use it to test whether a future, more stable selection protocol can recover the low-angle CFP win without handing the high-angle region to `step_selected`.
