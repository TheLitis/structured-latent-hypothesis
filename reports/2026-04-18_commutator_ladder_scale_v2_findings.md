# Commutator Ladder Scale V2 Findings - 2026-04-18

## Setup

- Benchmark family: `matched_scale_alpha`
- Operators: horizontal shift plus a brightness-anchored center-scale deformation
- Anchor: `alpha=0` now matches the repository `commutative` world exactly
- Split: `cartesian_blocks`
- Latent dimension: `4`
- Variants: `baseline`, `smooth`, `cfp_l0.010`, `cfp_l0.050`
- Seeds: `3, 11, 29`

## Main Result

This is now the strongest audited signal in the repository.

## What the Ladder Shows

- At `matched_scale_0.00`, baseline is best: `0.000884` vs `0.001031`.
- At `matched_scale_0.10`, CFP wins clearly:
  - baseline `0.001010`
  - `cfp_l0.010` `0.000910`
  - `cfp_l0.050` `0.000859`
- At `matched_scale_0.20`, CFP still wins:
  - baseline `0.002107`
  - `cfp_l0.010` `0.002046`
- At `matched_scale_0.35`, CFP gives the largest absolute gain in this ladder:
  - baseline `0.002991`
  - `cfp_l0.010` `0.002740`
- At `matched_scale_0.50`, the gain disappears and baseline is effectively tied or slightly better.

`smooth` does not explain the pattern. It is worse than baseline and worse than the winning CFP variant at every audited scale point.

## Interpretation

This family no longer says "CFP only helps at exactly zero commutator." The audited result is stronger and more interesting:

> CFP can help in a bounded band of mild-to-moderate non-commutativity, while still failing to dominate at the exact commuting anchor and at the strongest tested scale deformation.

That is a narrower claim than a universal CFP win, but it is also more substantive than the old exact-zero story.

## Research Decision

- Treat audited `scale_v2` as the current best evidence family.
- Before comparing with external popular baselines, add:
  1. `step_only` on this same ladder;
  2. a small nested lambda selection protocol;
  3. a latent-size sweep `2/4/8` only if the win persists under the first two checks.
