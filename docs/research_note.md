# Commutative Flatness Prior

## Positioning

This repository treats the original geometric construction as a starting point for a **research hypothesis**, not as an established law of optimization.

The central object is the mixed finite difference:

`Δ_i Δ_j z`

If it is small, the latent representation behaves locally like a sum of factor-specific shifts. In the exact discrete case, zero mixed difference implies an additive decomposition across indices.

## Core Hypothesis

If underlying factors are approximately independent and approximately commute, then regularizing latent representations toward small mixed finite differences should improve:

- compositional generalization,
- latent interpretability,
- robustness to unseen factor combinations.

## What Would Count As Real Evidence

- improvement on held-out combinations in synthetic commutative worlds,
- consistent reduction in latent commutativity error,
- a weaker or absent gain on non-commutative controls,
- stability across multiple seeds.

## What Would Not Count As Evidence

- a tiny improvement everywhere,
- gain that can be matched by generic smoothness,
- improved train reconstruction without better held-out composition,
- claims that CFP replaces dense layers or standard optimizers.

## Initial Experimental Program

### E0-A. Commutative Synthetic World

Use two independent factors whose order should not matter. The first version in this repository uses:

- horizontal translation,
- brightness scaling.

### E0-B. Non-Commutative Control

Use sequential factors where order sensitivity is structurally expected. The first version uses:

- asymmetric spatial windowing,
- rotation.

### Required Variants

- `baseline`: task loss only
- `smooth`: task loss plus generic curvature smoothing
- `comm`: task loss plus CFP mixed-difference penalty
- `comm_step`: CFP plus step-consistency penalty

## Interpretation Rule

The interesting result is not “CFP helps a bit.” The interesting result is:

> CFP helps specifically when the world is close to additive and commutative.

That is the threshold this repository is designed to test.
