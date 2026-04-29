# Support Routing Cost Robustness v1

## Result

- Best cost-aware family at 3 calibration worlds: `validation_loss` with `4/5` profiles passed.
- Validation-loss pass count at 3 worlds: `4/5`.
- Conformal-rank pass count at 3 worlds: `4/5`.
- Router-margin pass count at 3 worlds: `4/5`.
- Support-commutator reference pass count at 3 worlds: `1/5`.
- Validation-loss pass count at 5 worlds: `5/5`.
- Expensive-escalation validation-loss delta at 3 worlds: `+0.462798`.
- Expensive-escalation validation-loss delta at 5 worlds: `-0.022917`.

## Interpretation

This probe tests the new project claim: support-calibrated routing is useful only if it remains better than trivial actions under multiple deployment cost models.

The pivot survives this first gate: simple support-validation and conformal-rank policies are robust across most cost profiles with only 2-3 calibrated worlds and across all tested profiles with 5 calibrated worlds.

The hard regime is expensive escalation. With only 3 calibrated worlds, validation-loss still loses there; at 5 calibrated worlds it barely clears the gate. That makes calibration sample size the next bottleneck.

The support-commutator reference remains weak, so the active project should not re-center the three-point geometry.

Cost-aware training is the deployable upper bound when the deployment cost model is known. Default-trained evaluation tests whether one fixed policy can survive cost shift.
