# Transport Pair Probe Findings - 2026-04-18

## Setup

- World pairs:
  - `stepcurve_gamma`: separable transport with nonlinear step drift and zero commutator
  - `stepcurve_path_gamma`: matched path-dependent sibling with the same step drift but non-zero commutator
- Matched drift levels: `gamma = 1.00`, `2.00`, `4.00`
- Split: `cartesian_blocks`
- Selection: nested inner validation
- Variants:
  - `baseline`
  - `smooth`
  - fixed `cfp_l0.010`
  - fixed `affine_l0.010_s0.005`
  - `step_selected`
  - `cfp_selected`
  - `affine_selected`

## Main Result

This is the first probe in the repository that gives a clean, matched confirmation of the separable-transport hypothesis.

## Decisive Pattern

At matched step drift, the sign of the CFP gain tracks separability rather than equal spacing.

### `gamma = 1.00`

- Separable world:
  - baseline `0.001053`
  - `cfp_l0.010` `0.000984`
- Path-dependent sibling:
  - baseline `0.001085`
  - `cfp_l0.010` `0.001127`

### `gamma = 2.00`

- Separable world:
  - baseline `0.000994`
  - `cfp_l0.010` `0.000932`
- Path-dependent sibling:
  - baseline `0.000724`
  - `cfp_l0.010` `0.000793`

### `gamma = 4.00`

- Separable world:
  - baseline `0.001100`
  - `cfp_l0.010` `0.000768`
- Path-dependent sibling:
  - baseline `0.000967`
  - `cfp_l0.010` `0.000945`

Across all three matched pairs:

- fixed CFP helps on the separable world;
- the same CFP prior loses or nearly collapses on the path-dependent sibling;
- affine variants are consistently much worse than both baseline and CFP.

## Interpretation

This is exactly the interaction we wanted:

> the useful signal follows path independence / separability, not equal-step affine structure.

The original triple-point math survives, but in its general theorem form:

`Delta_n Delta_k z ~= 0`

and the induced separable law

`z_{n,k} = a_n + b_k`,

not in the narrower affine form

`z_0 + n v + k u`.

## What Is Confirmed

Confirmed in the current synthetic suite:

- the affine-lattice hypothesis is false as a general ML bias;
- the separable-transport hypothesis is supported;
- CFP is valuable when it matches separable/path-independent structure, even under nonlinear axis warps;
- adding a hard stationary-step bias hurts rather than helps in these matched worlds.

## Scope

This is not yet a claim about all real ML tasks.

It is a confirmed synthetic research result: the transferable part of the original point geometry is separable transport, not equal spacing.
