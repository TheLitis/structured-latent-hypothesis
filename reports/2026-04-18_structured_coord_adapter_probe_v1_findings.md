# Structured Coordinate Adapter Probe v1 Findings - 2026-04-18

## Setup

- World family: `stepcurve_coupled_4.00_alpha`
- Coupling ladder: `alpha = 0.00, 0.20, 0.35, 0.50, 0.75, 1.00`
- Split: `cartesian_blocks`
- Selection: nested inner validation
- New candidate:
  - `curv_hankel_coord_r4_selected`
- Comparison set:
  - `coord_latent`
  - `additive_resid_selected`
  - `curv_hankel_r4_selected`
  - `curvature_field_r4_selected`
  - `operator_diag_r2_selected`

This probe was meant to answer the most practical next question in the repository:

> can the triple-derived structured backbone become more engineering-useful
> by adding a small controlled coordinate adapter?

## Main Result

Not in the current form.

The structured-plus-coordinate adapter is a clean negative result.

It does **not** preserve the boundary gains of the structured hybrid, and it does **not** close the hard-tail gap to `coord_latent`.

So the simple recipe

> structured backbone + small coordinate pathway

is currently not enough.

## Decisive Pattern

### Exact separable anchor

- `alpha = 0.00`
  - `additive_resid_selected` `0.000452`
  - `operator_diag_r2_selected` `0.000500`
  - `curv_hankel_r4_selected` `0.000508`
  - `coord_latent` `0.000672`
  - `curv_hankel_coord_r4_selected` `0.000683`

The adapter already fails to improve even the easy anchor case.

### Boundary regime

- `alpha = 0.20`
  - `curv_hankel_r4_selected` `0.000694`
  - `additive_resid_selected` `0.000733`
  - `operator_diag_r2_selected` `0.000760`
  - `curvature_field_r4_selected` `0.000891`
  - `curv_hankel_coord_r4_selected` `0.000907`

- `alpha = 0.35`
  - `curv_hankel_r4_selected` `0.000863`
  - `operator_diag_r2_selected` `0.000968`
  - `coord_latent` `0.000980`
  - `additive_resid_selected` `0.000987`
  - `curv_hankel_coord_r4_selected` `0.002812`

At `alpha = 0.35`, where the pure structured hybrid was strongest, the coordinate-adapter model collapses badly.

This is the single clearest negative result in the cycle.

### Tail

- `alpha = 0.50`
  - `additive_resid_selected` `0.001051`
  - `curvature_field_r4_selected` `0.001164`
  - `coord_latent` `0.001176`
  - `curv_hankel_coord_r4_selected` `0.001352`

- `alpha = 0.75`
  - `coord_latent` `0.001127`
  - `curvature_field_r4_selected` `0.001315`
  - `additive_resid_selected` `0.001449`
  - `curv_hankel_coord_r4_selected` `0.001796`

- `alpha = 1.00`
  - `coord_latent` `0.001547`
  - `operator_diag_r2_selected` `0.001634`
  - `additive_resid_selected` `0.001638`
  - `curv_hankel_coord_r4_selected` `0.003041`

The adapter does not even help on the hard tail where extra flexibility was supposed to matter most.

## Interpretation

This probe rules out a tempting but too-simple path.

The repository now has evidence against three overly optimistic readings:

1. a fixed sum of structured components is enough across regimes;
2. a small generic coordinate adapter can repair that gap;
3. the remaining problem is just "not enough flexibility".

Instead, the issue looks more specific:

> the useful structured bias is regime-dependent,
> and naive extra flexibility destroys it rather than extending it.

That means the open problem is likely not a generic adapter.

It is either:

- explicit gating / regime selection,
- stronger optimization for structured models,
- or moving to a benchmark where the boundary-regime gains are the part that matters.

## What Is Now Supported

Supported in the current synthetic suite:

- the triple-derived structured backbone has real value in the boundary regime;
- naive coordinate-adapter augmentation does not preserve that value;
- the gap to practical usefulness is not solved by simply mixing in a little coordinate freedom.

## Research Consequence

This is a good stopping point for the current synthetic direction.

The next rational step is **not** another adapter variant.

It is one of two things:

1. a stricter adaptive/gated architecture with explicit regime selection;
2. a transfer benchmark outside the current synthetic family, to test whether boundary-regime gains are already the relevant object.

For practical value, the second option is now the stronger move.
