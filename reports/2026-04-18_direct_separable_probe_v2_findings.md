# Direct Separable Probe v2 Findings - 2026-04-18

## Setup

- Same matched world pairs as `direct_separable_probe_v1`:
  - `stepcurve_gamma`: separable transport with nonlinear step drift and zero commutator
  - `stepcurve_path_gamma`: matched path-dependent sibling with the same step drift but non-zero commutator
- Matched drift levels: `gamma = 1.00`, `2.00`, `4.00`
- Split: `cartesian_blocks`
- Selection: nested inner validation
- Variants:
  - `cell_latent`
  - `coord_latent`: a smooth non-factorized coordinate MLP baseline
  - `additive_only`
  - fixed `additive_resid_l0.010`
  - `additive_resid_selected`

## Main Result

The direct additive branch survives a stronger baseline.

It was not only beating a trivial per-cell lookup model.

The nested-selected additive-plus-residual model remains best across all six matched worlds, including against the new coordinate MLP baseline.

## Decisive Pattern

### `gamma = 1.00`

- Separable world:
  - `coord_latent` `0.000918`
  - `additive_only` `0.000820`
  - `additive_resid_selected` `0.000546`
- Path-dependent sibling:
  - `coord_latent` `0.000787`
  - `additive_only` `0.000804`
  - `additive_resid_selected` `0.000656`

### `gamma = 2.00`

- Separable world:
  - `coord_latent` `0.000925`
  - `additive_only` `0.000828`
  - `additive_resid_selected` `0.000637`
- Path-dependent sibling:
  - `coord_latent` `0.000805`
  - `additive_only` `0.001015`
  - `additive_resid_selected` `0.000654`

### `gamma = 4.00`

- Separable world:
  - `coord_latent` `0.000672`
  - `additive_only` `0.000779`
  - `additive_resid_selected` `0.000452`
- Path-dependent sibling:
  - `coord_latent` `0.000977`
  - `additive_only` `0.001209`
  - `additive_resid_selected` `0.000865`

## Interpretation

This changes the status of the direct branch.

- `cell_latent` is now clearly just a negative control.
- `coord_latent` is a real smooth generalization baseline.
- `additive_only` is specifically stronger on separable transport than on the matched path-dependent siblings.
- `additive_resid_selected` is the strongest model overall and consistently beats `coord_latent`.

So the additive parameterization is not a fake win caused by an unfair baseline.

## What This Also Reveals

The current path-dependent siblings are still too mild to cleanly separate:

- "small interaction correction on top of separability"
- from
- "genuinely richer non-separable structure".

Why: the selected additive-plus-residual model still wins on the path worlds.

That means the next decisive experiment should not be another lambda sweep.

It should be an **interaction ladder** with a controlled coupling strength, so that we can locate the point where a separable model stops being sufficient.

## Updated Working Claim

The strongest architectural reading of the project is now:

> useful structure is well captured by a separable latent backbone with a small interaction term,
> and the scientific question is where the regime boundary lies between "small correction"
> and "true non-separable coupling".
