# Transfer Policy Sweep Follow-Up

## Main Result

The explicit `structured / fallback / escalate` pipeline now has `14` cost configurations that beat the best trivial baseline simultaneously on `seed`, `world`, and `family` holdout.

The strongest found configuration is `structured_violation=5.0`, `fallback_overbudget=3.0`, `escalate_needed=0.5`, `escalate_unneeded=1.0`.

## Interpretation

This matters because it upgrades the previous claim. Before, the project only had a criterion that beat trivial baselines on some isolated tasks. Now the project also has an explicit decision pipeline with a nontrivial cost region where the policy survives `seed/world/family-out` validation.

## Limit

This is still not a universal law. The win depends on the deployment cost model, and it is anchored in the current synthetic context-transfer world. But it is stronger than a mere correlation claim and stronger than a single lucky threshold.

## Next Step

The next strict test should be cost robustness under unseen cost ratios or a semi-real transfer world, not a new model class.
