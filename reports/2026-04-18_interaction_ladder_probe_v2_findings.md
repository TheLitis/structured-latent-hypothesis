# Interaction Ladder Probe v2 Findings - 2026-04-18

## Setup

- World family: `stepcurve_coupled_4.00_alpha`
- Fixed nonlinear step drift: `gamma = 4.00`
- Controlled interaction ladder: `alpha = 0.00, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00`
- Split: `cartesian_blocks`
- Selection: nested inner validation
- Variants:
  - `coord_latent`
  - `additive_only`
  - fixed `additive_resid_l0.010`
  - `additive_resid_selected`

At `alpha = 0`, this ladder collapses exactly to the separable `stepcurve_4.00` world.

## Main Result

This is the first experiment in the repository that shows an explicit **regime boundary** for the new direct hypothesis.

The best bias is not universal.

Instead:

- a separable backbone with a small interaction correction is best in the low-to-moderate coupling regime;
- a more general coordinate model overtakes it once coupling becomes strong enough.

## Decisive Pattern

### Low to moderate coupling

- `alpha = 0.00`
  - `coord_latent` `0.000672`
  - `additive_resid_selected` `0.000452`
- `alpha = 0.10`
  - `coord_latent` `0.000664`
  - `additive_resid_selected` `0.000526`
- `alpha = 0.20`
  - `coord_latent` `0.000930`
  - `additive_resid_selected` `0.000733`

Here the separable-backbone model is clearly best.

### Boundary region

- `alpha = 0.35`
  - `coord_latent` `0.000980`
  - `additive_resid_selected` `0.000987`
- `alpha = 0.50`
  - `coord_latent` `0.001176`
  - `additive_resid_selected` `0.001051`

This is the crossover zone.

The direct additive branch is still competitive and often slightly better, but the margin has mostly collapsed.

### Strong coupling

- `alpha = 0.75`
  - `coord_latent` `0.001127`
  - `additive_resid_selected` `0.001449`
- `alpha = 1.00`
  - `coord_latent` `0.001547`
  - `additive_resid_selected` `0.001638`

At strong coupling, the more general coordinate model overtakes the separable-backbone model.

## Important Secondary Result

The nested selector does **not** respond to this transition by relaxing the residual penalty.

It chooses the strongest candidate in the current grid almost everywhere:

- `alpha = 0.00`: mostly `lambda_residual = 0.05`
- `alpha = 0.10`: mostly `lambda_residual = 0.05`
- `alpha >= 0.20`: always `lambda_residual = 0.05`

So the observed boundary is not a tuning artifact from the selector opening more residual capacity at large coupling.

It is a genuine mismatch between the separable-backbone bias and the strongest coupled worlds in this family.

## Interpretation

This sharpens the repository claim again.

The right working statement is now:

> separable transport is a real and useful inductive bias, but only inside a bounded interaction regime.

Or, architecturally:

> `z_{i,j} = a_i + b_j + r_{i,j}` is the right backbone in the low-coupling regime, but once interaction becomes too large, a more general non-factorized model wins.

That is a much stronger and more useful conclusion than the earlier universal affine-lattice idea.

## What Is Now Confirmed

Confirmed in the current synthetic suite:

- equal-step affine structure is not the right universal law;
- separable transport is the right abstraction at low interaction;
- direct additive parameterizations express that bias better than indirect AE-plus-penalty setups;
- there is a measurable coupling threshold beyond which the separable-backbone bias stops being optimal.

## Next Step

The next strongest experiment is no longer "does separability help at all?".

It is:

> what is the best explicit model class **beyond** the regime boundary?

That means introducing a structured interaction model, preferably low-rank or operator-based, and comparing it against `coord_latent` on the strong-coupling tail.
