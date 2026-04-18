# Transfer Law Budget Follow-Up

## What Changed

The previous cycle showed two things:

- triple-derived scores do track interaction and transfer risk;
- but the first router-style use of those scores did not beat `always_full`.

That left the project in an ambiguous state: diagnostically interesting, but not yet practically useful.

This follow-up asked a more concrete question:

> can triple-derived scores support cost-sensitive transfer decisions when the task is phrased as a budget or safety constraint, rather than as a direct model-choice router?

## Main Result

The answer is now:

> **partially yes.**

The scores still do not justify a universal router, but they now clear several explicit deployment-style tasks.

### Wins over trivial baselines

- `within_budget_6`: `diag_joint_prod` beats the best trivial baseline.
- `within_budget_8`: `diag_interaction` beats the best trivial baseline.
- `within_budget_10`: `diag_joint_sum` beats the best trivial baseline.
- `safe_regret_1e-05`: `diag_residual` beats the best trivial baseline with balanced accuracy `0.902`.

### Tasks that still fail

- `within_budget_4`
- `safe_regret_0`
- `safe_regret_5e-05`

So the useful regime is not “all budgets” or “all regret tolerances”. It is a narrower middle regime.

## New Interpretation

This is the first time the project has produced a criterion that is both:

- derived from the triple-point line of thought;
- and better than a trivial baseline under leave-one-seed-out validation.

That does **not** resurrect the original replacement-style claim.

What it supports instead is a narrower, stronger statement:

> triple-derived interaction geometry can support **budgeted transfer decisions** in a nontrivial subset of regimes.

This is materially stronger than “the score correlates with alpha”, and materially weaker than “the score can universally choose the right model.”

## Practical Meaning

The project now has its first realistic utility claim:

- not a new latent space,
- not a new optimizer,
- not a universal router,
- but a **risk / budget criterion** for deciding when transfer is likely cheap enough or when structured bias is safe enough.

That is the first branch that looks like it could matter outside the toy claim, because it can be evaluated as a decision rule rather than only as an explanatory score.

## Most Honest Next Step

The next iteration should stay in this lane and get stricter:

1. test leave-one-world-family-out instead of only leave-one-seed-out;
2. vary asymmetric error costs and check whether the win persists;
3. convert the best criterion into an explicit abstain / escalate policy:
   - use structured path if score says safe;
   - otherwise escalate adaptation budget or fall back.

If those stricter checks hold, the project finally has a credible “useful discovery” path:

> a transferable law for **budgeted adaptation and safe structured transfer**.
