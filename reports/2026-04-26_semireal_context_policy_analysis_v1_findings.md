# Semi-Real Context Policy Analysis v1

## Result

- Best in-domain split: `family` with delta `+0.562500`.
- Groups won: `0/3`.
- Best regret correlation: `score_joint_sum` = `-0.437`.

## Interpretation

This analysis checks whether the semi-real failure is only an external-threshold problem. If in-domain cross-validation also loses to trivial baselines, the next step must address the score representation itself, not just calibration.
