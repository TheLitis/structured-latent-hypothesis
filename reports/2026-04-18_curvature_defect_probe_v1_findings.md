# Curvature and Diagonal Defect Probe v1 Findings - 2026-04-18

## Setup

- World family: `stepcurve_coupled_4.00_alpha`
- Coupling ladder: `alpha = 0.00, 0.20, 0.35, 0.50, 0.75, 1.00`
- Split: `cartesian_blocks`
- Selection: nested inner validation
- New triple-derived candidates:
  - `curvature_field_r4_selected`
  - `hankel_r4_selected`
- Comparison baselines:
  - `coord_latent`
  - `additive_resid_selected`
  - `interaction_mlp_r4_selected`
  - `operator_diag_r2_selected`

This probe was designed to answer a sharper question than the earlier structured-tail runs:

> does the original triple-point geometry become more useful if interaction is parameterized
> directly through mixed difference or diagonal defect, rather than through free per-cell residuals?

## Main Result

Partially yes.

The new triple-derived models are **not** a universal replacement for the current baselines, but they are also clearly **not empty ideas**.

Two distinct signals appeared:

1. the **diagonal-defect / Hankel** model is the first direct descendant of the triple geometry that becomes the best overall model in the boundary regime;
2. the **curvature-field** model becomes the best structured model in part of the strong-coupling tail, even though it still does not beat the fully non-factorized `coord_latent`.

## Decisive Pattern

### Exact separable anchor

- `alpha = 0.00`
  - `additive_resid_selected` `0.000452`
  - `operator_diag_r2_selected` `0.000500`
  - `coord_latent` `0.000672`
  - `curvature_field_r4_selected` `0.000717`
  - `hankel_r4_selected` `0.000726`

At the exact separable anchor, the older additive-plus-residual branch still wins.

This is expected: once interaction vanishes, the extra curvature/diagonal structure is unnecessary.

### Boundary regime

- `alpha = 0.20`
  - `hankel_r4_selected` `0.000696`
  - `additive_resid_selected` `0.000733`
  - `operator_diag_r2_selected` `0.000760`
  - `interaction_mlp_r4_selected` `0.000801`
  - `curvature_field_r4_selected` `0.000891`
  - `coord_latent` `0.000930`

- `alpha = 0.35`
  - `hankel_r4_selected` `0.000962`
  - `operator_diag_r2_selected` `0.000968`
  - `coord_latent` `0.000980`
  - `additive_resid_selected` `0.000987`
  - `interaction_mlp_r4_selected` `0.001016`
  - `curvature_field_r4_selected` `0.001094`

This is the strongest positive result of the cycle.

The diagonal-defect model, which is the most literal continuation of the original triple-point diagonal law, becomes the best model at `alpha = 0.20` and `0.35`.

So the triple geometry was not exhausted by the earlier `additive_resid` branch.

### Late boundary / early tail

- `alpha = 0.50`
  - `additive_resid_selected` `0.001051`
  - `curvature_field_r4_selected` `0.001164`
  - `coord_latent` `0.001176`
  - `hankel_r4_selected` `0.001330`

The picture changes again.

Here the old additive-plus-residual branch is still best, but the curvature-field model is already better than `coord_latent` and clearly better than the diagonal-defect model.

This suggests that once coupling is no longer mostly diagonal, direct parameterization of `Delta_i Delta_j z` becomes more useful than diagonal defect.

### Strong tail

- `alpha = 0.75`
  - `coord_latent` `0.001127`
  - `curvature_field_r4_selected` `0.001315`
  - `interaction_mlp_r4_selected` `0.001372`
  - `additive_resid_selected` `0.001449`
  - `operator_diag_r2_selected` `0.001485`
  - `hankel_r4_selected` `0.001769`

- `alpha = 1.00`
  - `coord_latent` `0.001547`
  - `operator_diag_r2_selected` `0.001634`
  - `additive_resid_selected` `0.001638`
  - `interaction_mlp_r4_selected` `0.001958`
  - `curvature_field_r4_selected` `0.001995`
  - `hankel_r4_selected` `0.002188`

At `alpha = 0.75`, the curvature-field branch is the best structured model, but it still loses to `coord_latent`.

At `alpha = 1.00`, both new triple-derived candidates fail clearly, and the hard-tail winner remains the more general coordinate model.

## Interpretation

This probe changes the status of the triple-point program in an important way.

The honest conclusion is no longer:

> the original geometry only led to an over-restricted affine idea that failed.

It is now:

> the original geometry contains at least two nontrivial descendants:
> a diagonal-defect bias that helps in the boundary regime,
> and a curvature-field bias that improves part of the strong tail relative to older structured models.

So the core issue was not that the triple geometry was useless.

It was that the earlier parameterizations were still too indirect.

## What Is Now Supported

Supported in the current synthetic suite:

- direct parameterization of mixed difference or diagonal defect is more faithful to the original CFP mathematics than free cell residuals;
- the diagonal law of the triple construction is genuinely useful near the regime boundary;
- direct curvature-field parameterization helps on the early strong tail, even if it still does not close the final gap to `coord_latent`;
- neither of the new direct descendants supports a universal “replace the latent space” claim.

## Research Consequence

The next move is no longer "is the triple geometry empty?".

That question is now answered: no.

The sharper open question is:

> how should diagonal-defect and curvature-field structure be combined,
> or conditionally gated, so that the model keeps the boundary gains
> without losing the hard-tail expressivity that `coord_latent` still captures?

The most plausible next classes are:

- a hybrid `diagonal defect + curvature field` model;
- a gated hybrid that adds a controlled coordinate pathway on top of the structured transport backbone;
- or a transfer benchmark outside the current synthetic family, to see whether the new boundary gains are the part that actually matters.
