# Context Transfer Policy Sweep v1

## Result

- Tested `81` cost configurations.
- All-group winners: `14`.
- Best configuration: `svc=5.0`, `foc=3.0`, `enc=0.5`, `euc=1.0` with delta sum `-0.666667`.
- Best config group deltas: `seed=-0.273810`, `world=-0.285714`, `family=-0.107143`.

## Interpretation

This sweep tests whether the explicit decision pipeline only works at one hand-tuned cost setting or whether it has a real viable region. A positive result here means the policy has a nontrivial deployment regime, not just a lucky point estimate.
