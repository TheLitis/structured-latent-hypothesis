# Semi-Real Transfer Probe v1 Findings - 2026-04-18

## Setup

- World family: `semireal_coupled_alpha`
- Coupling ladder: `alpha = 0.00, 0.20, 0.35, 0.50, 0.75`
- Data: RGB scenes with textured asymmetric object, static background, cast shadow, fixed occluder, and scene-level sensor texture
- Split: `cartesian_blocks`
- Selection: nested inner validation
- Variants:
  - `coord_latent`
  - `additive_resid_selected`
  - `curv_hankel_r4_selected`
  - `curvature_field_r4_selected`
  - `operator_diag_r2_selected`

This probe was designed to answer the practical question that the synthetic ladder alone could not settle:

> do the triple-derived structured gains survive when the world becomes visually richer and less toy-like?

## Main Result

Not yet.

The semi-real benchmark is a clear reality check.

Across the whole tested ladder, the fully non-factorized `coord_latent` baseline is best or tied for best.

The triple-derived structured models remain competitive, but they do **not** become the leading solution.

## Decisive Pattern

### Exact separable anchor

- `alpha = 0.00`
  - `coord_latent` `0.000125`
  - `operator_diag_r2_selected` `0.000128`
  - `curv_hankel_r4_selected` `0.000173`
  - `curvature_field_r4_selected` `0.000180`
  - `additive_resid_selected` `0.000190`

Even at the exact separable anchor, the rich coordinate baseline already wins.

This is different from the earlier toy-world ladders, where structured models could dominate at low interaction.

### Mild to moderate coupling

- `alpha = 0.20`
  - `coord_latent` `0.000120`
  - `operator_diag_r2_selected` `0.000140`
  - `curv_hankel_r4_selected` `0.000166`
  - `curvature_field_r4_selected` `0.000169`
  - `additive_resid_selected` `0.000197`

- `alpha = 0.35`
  - `coord_latent` `0.000118`
  - `operator_diag_r2_selected` `0.000141`
  - `curv_hankel_r4_selected` `0.000167`
  - `curvature_field_r4_selected` `0.000180`
  - `additive_resid_selected` `0.000200`

This is the most important part of the result.

The structured hierarchy is still visible:

- `operator_diag` is the best structured model;
- `curv_hankel` is the next best;
- `curvature_field` follows;
- `additive_resid` is weakest.

So the triple-derived ideas are not random noise.

But the absolute winner is still the more general coordinate model.

### Stronger coupling

- `alpha = 0.50`
  - `coord_latent` `0.000124`
  - `operator_diag_r2_selected` `0.000156`
  - `curv_hankel_r4_selected` `0.000184`
  - `curvature_field_r4_selected` `0.000184`
  - `additive_resid_selected` `0.000207`

- `alpha = 0.75`
  - `coord_latent` `0.000154`
  - `operator_diag_r2_selected` `0.000155`
  - `curv_hankel_r4_selected` `0.000180`
  - `curvature_field_r4_selected` `0.000187`
  - `additive_resid_selected` `0.000226`

At `alpha = 0.75`, `operator_diag` comes very close to `coord_latent`, but it still does not overtake it.

So the best current reading is:

> structured models are competitive on semi-real scenes,
> but not yet strong enough to claim practical superiority.

## Interpretation

This probe changes the status of the whole project in a useful way.

Before it, the main uncertainty was whether the synthetic gains were merely toy artifacts.

Now the picture is sharper:

- the structured hierarchy does survive qualitatively into a richer visual setting;
- but the performance advantage does not survive quantitatively.

That means the project still has research value, but not yet the kind of result that would make a large lab care on engineering grounds.

## What Is Now Supported

Supported in the current repository:

- triple-derived structure is a real source of inductive bias on synthetic ladders;
- some of that hierarchy survives on a richer, semi-real visual benchmark;
- however, the current structured models do not outperform a strong smooth coordinate baseline once the visuals become more realistic.

## Research Consequence

This is the first point where the project has a clean externality test, and it fails to deliver a practical win.

So the next move should **not** be more internal synthetic elaboration.

The strongest options now are:

1. turn to optimization/training geometry, where the triple-point math may still open a different path;
2. design a much stronger regime-gated structured model, but only if it is clearly different from the failed adapter path;
3. or pause the "large-company relevance" claim until a benchmark with real practical gain exists.

The honest conclusion after this probe is:

> the idea is scientifically alive, but not yet practically validated.
