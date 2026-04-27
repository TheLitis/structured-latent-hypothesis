# Support Contrast Calibration Follow-Up

## Main Result

Label-free rank calibration does not win on synthetic-to-semi-real transfer.

Small target rank calibration wins `2/3` semi-real CV splits.

## Interpretation

If rank external fails but target-rank CV wins, the criterion needs target-domain calibration labels. If both fail, support contrast is not enough for the current semi-real benchmark.
