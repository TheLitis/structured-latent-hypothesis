# Direct Separable Probe Findings - 2026-04-18

## Setup

- World pairs:
  - `stepcurve_gamma`: separable transport with nonlinear step drift and zero commutator
  - `stepcurve_path_gamma`: matched path-dependent sibling with the same step drift but non-zero commutator
- Matched drift levels: `gamma = 1.00`, `2.00`, `4.00`
- Split: `cartesian_blocks`
- Variants:
  - `cell_latent`: one free latent per cell plus a shared decoder
  - `additive_only`: `z_{i,j} = a_i + b_j`
  - `additive_resid_l0.010`: `z_{i,j} = a_i + b_j + r_{i,j}` with fixed residual penalty
  - `additive_resid_selected`: the same residual model with nested lambda selection

## Main Result

The direct additive parameterization is materially stronger than the old "autoencoder plus CFP penalty" route as a way to express the separable-transport hypothesis.

## Decisive Pattern

The factorized additive model already captures most of the compositional signal, while the fully free per-cell latent baseline fails badly on holdout cells.

### `gamma = 1.00`

- Separable world:
  - `cell_latent` `0.022825`
  - `additive_only` `0.000820`
  - `additive_resid_selected` `0.000546`
- Path-dependent sibling:
  - `cell_latent` `0.026928`
  - `additive_only` `0.000804`
  - `additive_resid_selected` `0.000656`

### `gamma = 2.00`

- Separable world:
  - `cell_latent` `0.018120`
  - `additive_only` `0.000828`
  - `additive_resid_selected` `0.000637`
- Path-dependent sibling:
  - `cell_latent` `0.019124`
  - `additive_only` `0.001015`
  - `additive_resid_selected` `0.000654`

### `gamma = 4.00`

- Separable world:
  - `cell_latent` `0.016300`
  - `additive_only` `0.000779`
  - `additive_resid_selected` `0.000452`
- Path-dependent sibling:
  - `cell_latent` `0.015546`
  - `additive_only` `0.001209`
  - `additive_resid_selected` `0.000865`

Across all three matched pairs:

- `cell_latent` is not a viable compositional model and should be treated as a negative control only;
- `additive_only` is already strong and systematically better on separable worlds than on matched path-dependent siblings as drift grows;
- `additive_resid_selected` is the best-performing direct model in this probe and preserves the same separable-vs-path ordering.

## Interpretation

This result sharpens the project direction.

The core useful object is no longer "CFP as a penalty attached to a generic latent model".

It is a direct additive latent structure:

`z_{i,j} = a_i + b_j + r_{i,j}`

where most of the useful signal sits in the additive term and the residual term acts as a small interaction correction.

## Important Limitation

This probe does **not** yet show that residual magnitude is a clean path-dependence estimator.

Because `r_{i,j}` is still a free per-cell interaction tensor, the nested-selected model can improve reconstruction while keeping the residual norm very small on held-out cells. That means the reconstruction gap is currently more informative than the raw residual magnitude.

## What This Changes

The new strongest working branch of the repository is:

> separable transport should be tested with direct factorized parameterizations first, and only secondarily with indirect flatness penalties on a generic autoencoder.

This does not replace the confirmed separable-transport hypothesis.

It refines how the hypothesis should be expressed architecturally.
