# Support Contrast Calibration Probe v1

## Result

- Raw external delta: `+0.700000`.
- Rank external delta: `+1.700000`.
- Target-rank CV wins: `2/3`.
- Hybrid-rank CV wins: `0/3`.

## Interpretation

This probe checks whether support-contrast failure is caused by score scale mismatch. Rank external is label-free target-domain calibration; target-rank CV is small labeled target calibration; hybrid-rank CV adds synthetic rows back into the calibrated target policy.
