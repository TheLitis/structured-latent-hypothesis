# Context Transfer Cost Shift v1

## Result

- Source policies tested: `14` prior all-group winners.
- Unseen target cost configs: `24`.
- All-target robust source policies: `0`.
- Prior best source wins `11/24` target configs; worst group delta `+0.196429`.
- Best robust source wins `11/24` target configs with worst group delta `+0.196429`.

## Interpretation

This probe separates cost selection from cost evaluation. The strong cost-robust claim does **not** pass: the best source policy still wins on `11/24` target settings, but no source policy survives every unseen target cost and every `seed/world/family` split.

The narrower conclusion is that the decision law remains useful only when the deployment cost model is known closely enough to calibrate the policy. That is still deployable in principle, but it is not a cost-invariant transfer law.
