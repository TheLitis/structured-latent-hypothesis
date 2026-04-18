# Low-Rank Tail Probe v1 Findings - 2026-04-18

## Setup

- World family: `stepcurve_coupled_4.00_alpha`
- Boundary and strong-coupling tail:
  - `alpha = 0.35, 0.50, 0.75, 1.00`
- Split: `cartesian_blocks`
- Variants:
  - `coord_latent`
  - `additive_resid_selected`
  - `lowrank_r1_l0.050`
  - `lowrank_r2_l0.050`
  - `lowrank_r4_l0.050`
  - `lowrank_r4_selected`

The low-rank interaction term is centered so it cannot absorb row-only or column-only main effects.

## Main Result

Low-rank interaction extends the separable backbone slightly past the clean separable regime, but it does **not** solve the strong-coupling tail.

It is a meaningful intermediate model, not the final answer.

## Decisive Pattern

### Boundary point: `alpha = 0.35`

- `lowrank_r2_l0.050` `0.000850`
- `coord_latent` `0.000980`
- `additive_resid_selected` `0.000987`

At the boundary, low-rank interaction helps.

### Near-boundary point: `alpha = 0.50`

- `additive_resid_selected` `0.001051`
- `lowrank_r2_l0.050` `0.001078`
- `coord_latent` `0.001176`

Here low-rank interaction remains competitive and still beats the coordinate baseline.

### Strong tail: `alpha = 0.75`

- `coord_latent` `0.001127`
- `additive_resid_selected` `0.001449`
- `lowrank_r4_selected` `0.001532`
- `lowrank_r2_l0.050` `0.001540`

### Strong tail: `alpha = 1.00`

- `coord_latent` `0.001547`
- `additive_resid_selected` `0.001638`
- `lowrank_r4_selected` `0.001928`
- `lowrank_r2_l0.050` `0.002045`

At strong coupling, every tested low-rank variant loses.

## Important Secondary Result

Tuning does not rescue the low-rank branch.

For `lowrank_r4_selected`, the nested selector mostly chooses moderate or strong regularization:

- `alpha = 0.35`: mostly `lambda_residual = 0.02`
- `alpha = 0.50`: always `0.05`
- `alpha = 0.75`: mostly `0.05`
- `alpha = 1.00`: mostly `0.05`

So the failure on the strong tail is not explained by an obviously bad regularization setting.

## Interpretation

This result sharpens the modeling picture:

- pure separable structure works in the low-coupling regime;
- low-rank interaction extends that regime a bit further;
- but strong coupling in this family is not well captured by a small centered bilinear interaction.

That means the next model class should not just be "more of the same" low-rank factorization.

It should be a qualitatively richer interaction mechanism, likely:

- a non-factorized coordinate model, which already wins on the strong tail, or
- an operator-based interaction model, if we want a more interpretable successor to separable transport.

## Updated Working Claim

The repository now supports a three-regime picture:

1. low coupling: separable backbone is best;
2. boundary region: low-rank interaction can help;
3. strong coupling: general non-factorized structure wins.
