# CFP Sweep V1 Findings - 2026-04-18

## Setup

- Split: `cartesian_blocks`
- Worlds: `commutative`, `noncommutative`
- Latent dimension: `4`
- Variants: `baseline`, `smooth`, `cfp_l0.010`, `cfp_l0.020`, `cfp_l0.050`, `cfp_l0.100`, `cfp_l0.180`
- Seeds: `3, 11, 29`

## Main Outcome

This sweep does **not** support the narrow claim that CFP helps specifically on the commutative world.

## What Happened

### Commutative World

- Best CFP variant: `cfp_l0.010`
- Baseline held-out reconstruction: `0.000884`
- Best CFP held-out reconstruction: `0.001031`
- Baseline holdout-cell comm error: `0.003367`
- Best CFP holdout-cell comm error: `0.002473`

Interpretation: CFP makes the latent flatter on held-out cells, but the gain is not translating into better held-out prediction quality.

### Non-Commutative World

- Best CFP variant: `cfp_l0.010`
- Baseline held-out reconstruction: `0.002599`
- Best CFP held-out reconstruction: `0.002447`
- Baseline holdout-cell comm error: `0.013942`
- Best CFP holdout-cell comm error: `0.017459`

Interpretation: the current non-commutative control is not clean enough. CFP slightly improves held-out reconstruction here while worsening holdout-cell flatness, which suggests the benchmark is confounding commutativity with other forms of regularization or representation pressure.

## Research Decision

Do not spend more time tuning `lambda_comm` on this exact benchmark family.

The next benchmark should change one thing:

1. keep the operator family matched,
2. vary the true commutator magnitude in a controlled ladder,
3. measure whether CFP gain tracks that ground-truth commutator level.

That is a stronger test of the actual claim than more hyperparameter search on the current pair of worlds.
