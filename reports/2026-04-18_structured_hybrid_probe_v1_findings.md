# Structured Hybrid Probe v1 Findings - 2026-04-18

## Setup

- World family: `stepcurve_coupled_4.00_alpha`
- Coupling ladder: `alpha = 0.00, 0.20, 0.35, 0.50, 0.75, 1.00`
- Split: `cartesian_blocks`
- Selection: nested inner validation
- New candidate:
  - `curv_hankel_r4_selected`
- Comparison set:
  - `coord_latent`
  - `additive_resid_selected`
  - `curvature_field_r4_selected`
  - `hankel_r4_selected`
  - `operator_diag_r2_selected`

This probe was designed to test the most direct next question after the curvature/defect cycle:

> can a single hybrid structured model combine the boundary gains of diagonal defect
> with the tail gains of curvature field?

## Main Result

Not fully.

But the failure mode is informative rather than disappointing.

The simple hybrid does **not** unify the whole coupling ladder.
Instead, it becomes the strongest model exactly in the boundary regime, and then degrades again on the strong tail.

So the right updated statement is:

> the two triple-derived structures are complementary,
> but they cannot simply be added together and expected to solve all regimes.

## Decisive Pattern

### Exact separable anchor

- `alpha = 0.00`
  - `additive_resid_selected` `0.000452`
  - `operator_diag_r2_selected` `0.000500`
  - `curv_hankel_r4_selected` `0.000508`
  - `coord_latent` `0.000672`

The hybrid stays close to the best structured models, but it does not become the new anchor winner.

### Boundary regime

- `alpha = 0.20`
  - `curv_hankel_r4_selected` `0.000694`
  - `hankel_r4_selected` `0.000696`
  - `additive_resid_selected` `0.000733`
  - `coord_latent` `0.000930`

- `alpha = 0.35`
  - `curv_hankel_r4_selected` `0.000863`
  - `hankel_r4_selected` `0.000962`
  - `operator_diag_r2_selected` `0.000968`
  - `coord_latent` `0.000980`
  - `additive_resid_selected` `0.000987`

This is the strongest result of the probe.

At `alpha = 0.35`, the hybrid is not only the best structured model, but the best model overall with a clear margin.

So combining the two triple-derived structures is useful, but specifically in the moderate-interaction boundary zone.

### Late boundary / early tail

- `alpha = 0.50`
  - `additive_resid_selected` `0.001051`
  - `curvature_field_r4_selected` `0.001164`
  - `coord_latent` `0.001176`
  - `curv_hankel_r4_selected` `0.001248`

At `alpha = 0.50`, the hybrid already loses the plot.

The direct additive-plus-residual branch is again best, and the hybrid falls behind both `curvature_field` and `coord_latent`.

### Strong tail

- `alpha = 0.75`
  - `coord_latent` `0.001127`
  - `curvature_field_r4_selected` `0.001315`
  - `additive_resid_selected` `0.001449`
  - `operator_diag_r2_selected` `0.001485`
  - `curv_hankel_r4_selected` `0.001606`

- `alpha = 1.00`
  - `coord_latent` `0.001547`
  - `operator_diag_r2_selected` `0.001634`
  - `additive_resid_selected` `0.001638`
  - `curv_hankel_r4_selected` `0.001954`

On the strong tail, the simple hybrid is clearly worse than both the best structured baselines and the non-factorized coordinate model.

## Interpretation

This probe gives a much sharper structural lesson.

The triple-point descendants do not combine by naive superposition.

Why this likely happens:

- diagonal defect is useful when interaction remains aligned with the triple-law diagonals;
- curvature field is useful when path dependence is better described as distributed local mixed difference;
- once both are present strongly, a fixed additive sum over-commits to structure instead of adapting.

So the remaining gap is not just "more structure".

It is **adaptive structure selection**.

## What Is Now Supported

Supported in the current synthetic suite:

- direct triple-derived structure is real, not empty;
- a simple curvature-plus-diagonal hybrid is strongest in the boundary regime;
- the same hybrid is not sufficient on the strong tail;
- therefore the next useful class must be adaptive rather than a fixed sum of structured components.

## Research Consequence

The next strongest move is no longer another pure structured sum.

It is one of:

- a gated structured model that can turn curvature or diagonal channels up and down by regime;
- a structured backbone plus a **controlled coordinate adapter**;
- or a transfer benchmark showing that the boundary regime is the practically relevant one.

For practical relevance, the most promising immediate next step is the second option:

> structured backbone + small coordinate adapter

because it directly tests whether the triple-derived structure can survive inside a more expressive, engineering-useful architecture.
