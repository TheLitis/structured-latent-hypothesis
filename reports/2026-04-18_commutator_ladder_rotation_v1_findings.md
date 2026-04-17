# Commutator Ladder Rotation V1 Findings - 2026-04-18

## Setup

- Benchmark family: `matched_rotate_alpha`
- Operators: horizontal shift plus a brightness-anchored small-rotation deformation
- Anchor: `alpha=0` matches the repository `commutative` world exactly
- Split: `cartesian_blocks`
- Latent dimension: `4`
- Variants: `baseline`, `smooth`, `cfp_l0.010`, `cfp_l0.050`
- Seeds: `3, 11, 29`

## Main Result

The first audited rotation ladder is mixed, but it is not purely negative.

## What Happened

- At `matched_rotate_0.00`, baseline is best: `0.000884` vs `0.001031`.
- At `matched_rotate_5.00`, CFP wins clearly:
  - baseline `0.001071`
  - `cfp_l0.010` `0.000895`
  - `cfp_l0.050` `0.000849`
- At `matched_rotate_10.00` and `20.00`, baseline is better than both CFP variants.
- At `matched_rotate_30.00`, `cfp_l0.010` gets a very small edge: `0.002938` vs `0.002992`.

Again, `smooth` is not competitive and does not explain the effect.

## Interpretation

The rotation family does not yet show a clean monotone CFP advantage window. The signal is real enough to keep, but still unstable:

> CFP helps at very small rotation and barely helps again at the strongest tested angle, while losing in the middle of the ladder.

That pattern is more consistent with lambda mismatch or family-specific heterogeneity than with a simple "smaller commutator is always better for CFP" law.

## Research Decision

- Keep rotation as a live benchmark family.
- The next meaningful move here is not a broader claim but a better hyperparameter protocol:
  1. add `step_only`;
  2. replace the hand-picked CFP lambdas with a small nested selection;
  3. only then decide whether the rotation family is genuinely supportive or just noisy.
