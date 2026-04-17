# Matched Family Benchmark Audit - 2026-04-18

## Scope

Audit the synthetic matched-family benchmark before using it as evidence for or against CFP.

## Findings

- `matched_comm` is internally consistent. Its world generator and its reported ground-truth commutator both correspond to the same operator pair: horizontal shift plus position-dependent intensity modulation.
- The original `matched_scale_v1` was not a valid matched commutator ladder. The generator pre-scaled the base image once and then sampled a `shift x brightness` grid, while the reported commutator metadata referred to a different `shift x center-scale` story.
- The old matched ladders also used a slightly different brightness range from the repository's `commutative` world, so `alpha=0` was not a shared anchor across all families.

## Fixes Applied

- Rewrote the scale family so the column operator deforms a commuting brightness transform into a brightness-anchored center-scale transform as `alpha` grows.
- Added a rotation family with the same contract: `alpha=0` is the shared commuting brightness operator, and `alpha>0` injects a brightness-anchored rotation.
- Aligned the matched families to the same `commutative` anchor at `alpha=0`.
- Added tests that assert all three matched families reduce exactly to the same `commutative` world at zero strength and that their ground-truth commutator metadata grows monotonically.

## Research Consequence

- Treat `commutator_ladder_scale_v1` as superseded.
- Prefer the audited set for any cross-family claim:
  - `commutator_ladder_ramp_v2`
  - `commutator_ladder_scale_v2`
  - `commutator_ladder_rotation_v1`
