# Commutator Ladder Ramp V2 Findings - 2026-04-18

## Setup

- Benchmark family: `matched_comm_alpha`
- Operators: horizontal shift plus position-dependent intensity modulation
- Anchor: `alpha=0` now matches the repository `commutative` world exactly
- Split: `cartesian_blocks`
- Latent dimension: `4`
- Variants: `baseline`, `smooth`, `cfp_l0.010`, `cfp_l0.050`
- Seeds: `3, 11, 29`

## Main Result

After the benchmark audit and the shared zero-point anchor fix, the earlier near-zero CFP gain disappears in this family.

## What Happened

- At `matched_comm_0.00`, baseline is best: `0.000884` vs `0.001031` for `cfp_l0.010`.
- At `matched_comm_0.10`, baseline is still best: `0.001350` vs `0.001396`.
- At `matched_comm_0.20`, `cfp_l0.010` gets a small win: `0.001396` vs `0.001473`.
- At `matched_comm_0.35` and `0.50`, baseline wins again.

`smooth` is clearly not the explanation here: it is much worse than both baseline and CFP everywhere in this audited ramp ladder.

## Interpretation

This family no longer supports a clean "CFP helps near the commuting point" story.

The more defensible reading is:

> in the audited ramp family, CFP is not broadly helpful; there is one mild mid-regime gain, but baseline dominates at the exact commuting anchor and at higher commutator strengths.

## Research Decision

- Keep the ramp ladder as a negative or mixed control, not as the headline result.
- Do not use the earlier ramp-v1 pattern as the primary evidence claim anymore.
