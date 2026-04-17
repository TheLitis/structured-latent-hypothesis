# Step Stationarity Probe Findings - 2026-04-18

## Setup

- World family: `stepcurve_gamma`
- Shared property: all worlds have zero ground-truth commutator
- Controlled variable: step drift along the brightness axis
- `gamma = 1.00` is the exact affine/stationary-step case
- `gamma > 1.00` preserves commutativity but breaks equal spacing
- Variants:
  - `baseline`
  - `smooth`
  - `cfp_l0.010`
  - `affine_l0.010_s0.005`
  - `step_selected`
  - `cfp_selected`
  - `affine_selected`

## Main Result

The affine-lattice hypothesis does not survive this probe.

## What Happened

- At the exact affine point `stepcurve_1.00`, the best result is already `cfp_l0.010`, not the affine prior.
- As step drift increases, fixed CFP stays competitive and then becomes clearly strongest:
  - `stepcurve_2.00`: `cfp_l0.010` beats baseline, while `affine_selected` is much worse.
  - `stepcurve_3.00`: `cfp_l0.010` gives the best result in the whole table.
  - `stepcurve_4.00`: both `cfp_l0.010` and `cfp_selected` beat baseline, while both affine variants remain clearly worse.
- `step_selected` and `affine_selected` strongly reduce holdout-cell comm error, but that extra step regularization does not translate into better held-out reconstruction.

## Interpretation

This is a clean falsification of the stronger "equal spacing is the transferable law" story.

The data say:

> commutative/separable structure can still be useful even when step stationarity is broken, and forcing a global affine-lattice latent can actively hurt.

That means the part of the original point geometry that transfers is not the equal spacing itself.

## Research Consequence

The next working hypothesis should move from

> `z = z0 + n v + k u`

to the more general separable law

> `z = a_n + b_k`

with nonlinear coordinate warps allowed along each factor axis.
