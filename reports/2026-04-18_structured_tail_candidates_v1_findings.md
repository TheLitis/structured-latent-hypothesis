# Structured Tail Candidates v1 Findings - 2026-04-18

## Setup

- World family: `stepcurve_coupled_4.00_alpha`
- Strong-coupling region:
  - `alpha = 0.50, 0.75, 1.00`
- Split: `cartesian_blocks`
- Variants:
  - `coord_latent`
  - `additive_resid_selected`
  - `lowrank_r4_selected`
  - `interaction_mlp_r4_selected`
  - `operator_full_r2_selected`
  - `operator_diag_r2_selected`

This probe was designed to answer a narrow question:

> can a richer structured interaction model beat the fully non-factorized coordinate baseline on the strong-coupling tail?

## Main Result

Not yet.

No tested structured successor overtakes `coord_latent` on the strong-coupling tail.

However, the richer structured models are not all equivalent:

- they do improve over the earlier low-rank branch on part of the tail;
- they still fail to close the final gap to the coordinate baseline.

## Decisive Pattern

### `alpha = 0.50`

- `additive_resid_selected` `0.001051`
- `lowrank_r4_selected` `0.001155`
- `coord_latent` `0.001176`
- `operator_full_r2_selected` `0.001277`

The tail has not fully started yet. The old additive-plus-residual branch is still best here.

### `alpha = 0.75`

- `coord_latent` `0.001127`
- `interaction_mlp_r4_selected` `0.001372`
- `operator_full_r2_selected` `0.001425`
- `additive_resid_selected` `0.001449`
- `lowrank_r4_selected` `0.001532`

At this point, richer structured interaction helps relative to the older structured branches, but still loses to the coordinate baseline.

### `alpha = 1.00`

- `coord_latent` `0.001547`
- `operator_diag_r2_selected` `0.001634`
- `additive_resid_selected` `0.001638`
- `operator_full_r2_selected` `0.001916`
- `lowrank_r4_selected` `0.001928`
- `interaction_mlp_r4_selected` `0.001958`

At the strongest tested coupling, the diagonal transport operator is the best structured candidate, but it still does not beat `coord_latent`.

## Interpretation

This probe changes the tail story in an important way.

The conclusion is no longer:

> low-rank fails, therefore structure is exhausted.

It is now:

> stronger structured interaction models can recover some of the lost ground,
> but the current structured candidates still do not match a fully non-factorized coordinate model on the strongest coupled worlds.

That is a cleaner and more useful negative result.

## What This Means

The repository now supports the following picture:

1. low coupling:
   separable backbone is best;
2. boundary region:
   low-rank interaction helps;
3. early strong tail:
   richer structured interaction improves over low-rank;
4. hard tail:
   `coord_latent` still wins.

## Research Consequence

The open problem is now sharper.

It is not just "add more interaction".

It is:

> what structured model class can preserve interpretability while matching the expressivity of `coord_latent` on the hard tail?

The most plausible next directions are:

- a hybrid model that adds a controlled coordinate pathway to the structured backbone;
- an operator model with deeper composition rather than a single residual transport step;
- or a transfer test outside the current synthetic family to check whether the structured gap matters in more realistic settings.
