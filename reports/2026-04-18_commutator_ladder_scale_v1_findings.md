# Commutator Ladder Scale V1 Findings - 2026-04-18

## Setup

- Benchmark family: `matched_scale_alpha`
- Operators: horizontal shift + center scaling
- Split: `cartesian_blocks`
- Latent dimension: `4`
- Variants: `baseline`, `smooth`, `cfp_l0.010`, `cfp_l0.050`
- Seeds: `3, 11, 29`

## Main Result

This second matched family gives a **stricter** result than the first ladder.

## What Happened

### Exact commuting point

- At `matched_scale_0.00`, `cfp_l0.010` beats baseline: `0.000796` vs `0.000958`.

Interpretation: the mild CFP prior still helps when the world is exactly commuting.

### Non-zero commutator regimes

- At `matched_scale_0.10`, baseline is already slightly better than both CFP variants.
- At `matched_scale_0.20`, `0.35`, and `0.50`, baseline stays better than CFP.

Interpretation: in this family, the CFP benefit does not persist once the commutator becomes non-zero. The prior still lowers holdout-cell flatness, but that flatter latent geometry does not convert into better held-out prediction quality.

## Cross-Family Interpretation

Compared with the first ladder (`shift + ramp`), this weakens any broad claim that CFP generally helps in “approximately commuting” worlds.

The more defensible statement after two ladders is:

> a mild CFP prior can help in exactly commuting or extremely near-commuting settings, but the effect is not yet robust across matched operator families once non-commutativity is introduced.

## Research Decision

Do not claim broad support yet.

The next useful step is:

1. add a third matched family, preferably `translation + small rotation`;
2. then run a family-by-family summary comparing where the sign of the CFP gain flips;
3. only after that do capacity sweeps like latent size `2/4/8`.
