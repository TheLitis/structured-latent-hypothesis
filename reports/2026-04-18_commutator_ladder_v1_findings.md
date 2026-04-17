# Commutator Ladder V1 Findings - 2026-04-18

## Setup

- Benchmark family: `matched_comm_alpha`
- Operators: horizontal shift + position-dependent intensity ramp
- Split: `cartesian_blocks`
- Latent dimension: `4`
- Variants: `baseline`, `smooth`, `cfp_l0.010`, `cfp_l0.050`
- Seeds: `3, 11, 29`

## Main Result

This is the first benchmark in the repository that gives a **partially supportive** signal for CFP.

## What the Ladder Shows

### Exact and near-commutative regimes

- At `matched_comm_0.00`, `cfp_l0.010` beats baseline: `0.000794` vs `0.000972`.
- At `matched_comm_0.10`, `cfp_l0.010` also beats baseline: `0.000960` vs `0.001108`.

Interpretation: when the true commutator is zero or very small, a mild CFP prior can improve held-out composition performance.

### Mid regime

- At `matched_comm_0.20`, `cfp_l0.050` beats baseline: `0.000968` vs `0.001034`.

Interpretation: the best CFP strength may drift upward slightly as the world becomes less exactly additive, but the gain is already small.

### Higher non-commutative regimes

- At `matched_comm_0.35` and `matched_comm_0.50`, baseline is better than both CFP variants.

Interpretation: once the true commutator is large enough, the CFP inductive bias becomes a liability rather than an asset.

## Why This Matters

This is much closer to the claim we actually want to test:

> CFP helps in worlds that are close to commuting, and the benefit fades or reverses as true non-commutativity grows.

The current ladder does not fully prove that claim, but it is the first benchmark here that behaves in the right qualitative direction.

## Research Decision

Keep this matched-ladder family as a core benchmark.

Next steps should be:

1. repeat the ladder with a second matched operator family, preferably `translation + scale` or `translation + small rotation`;
2. sweep latent size `2/4/8` on this same ladder;
3. replace the ad-hoc best-lambda choice with a small nested selection protocol so the claimed gain is not tied to one manually chosen `lambda`.
