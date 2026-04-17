# Nested Selection Summary - 2026-04-18

## Main Takeaway

Adding `step_selected` and a nested lambda-selection protocol does not kill the CFP idea, but it narrows it sharply.

## What Survives

- The best current evidence is still the audited `scale` family.
- CFP remains strongest at mild-to-moderate scale non-commutativity.
- `smooth` is not competitive, so the effect is still more specific than generic smoothing.
- The geometric part of the hypothesis still looks real: CFP keeps reducing commutativity error even when the task-level gain becomes unstable.

## What Weakens

- The nested CFP selector is unstable across seeds; it chooses widely different lambdas on the same family.
- `step_selected` becomes competitive or superior in some later-ladder regimes, especially `matched_scale_0.50` and `matched_rotate_30.00`.
- The rotation family still does not give a simple monotone CFP story.
- The practical performance claim is now narrower than the geometric claim: latent flattening survives more consistently than held-out reconstruction gain.

## Research Decision

The highest-value next experiment before any comparison with external popular baselines is:

1. keep the audited `scale` family as the main benchmark;
2. replace single-split nested selection with a more stable repeated inner-selection protocol;
3. then rerun `scale` and `rotation` with the same `step_selected` competitor and only after that compare against broader standard methods.
