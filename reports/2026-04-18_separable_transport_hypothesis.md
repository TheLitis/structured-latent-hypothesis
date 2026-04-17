# Separable Transport Hypothesis - 2026-04-18

## Why the Hypothesis Changed

The original triple-point picture suggested the exact affine form

`z_{n,k} = z_0 + n v + k u`.

That contains two distinct assumptions:

1. factor composition is separable and commutative;
2. the steps along each factor axis are stationary and equally spaced.

The new step-stationarity probe shows that assumption 2 is too strong for the transferable ML claim.

## New Working Hypothesis

The right generalization is the separable form already implied by zero mixed difference:

`z_{n,k} = a_n + b_k`.

Equivalent statement:

`Delta_n Delta_k z ~= 0`

is the core law, while the affine special case

`a_n = n v, b_k = k u`

is only one coordinate choice.

## Interpretation

This makes the original point geometry more important, not less.

The breakthrough is no longer:

> useful latents must be affine lattices.

It becomes:

> useful latents may admit factor-wise additive transport even under nonlinear reparameterization of each axis.

That is a stronger conceptual claim because it is more invariant:

- it preserves path independence;
- it does not require equal speed along the factor axes;
- it fits the theorem `Delta_n Delta_k z = 0 => z = a_n + b_k`;
- it survives the new probe where equal spacing was broken on purpose.

## Practical Consequence

CFP still looks relevant, but now as a path-independence or separable-transport prior rather than an affine-grid prior.

The next experiments should therefore focus on methods that preserve separability under axis warps, not on stronger global step-equality penalties.
