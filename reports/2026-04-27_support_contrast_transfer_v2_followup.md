# Support Contrast Transfer Probe v2 Follow-Up

## Main Result

The support-contrast criterion does not beat the best trivial baseline when trained on synthetic context-transfer and tested on semi-real context-transfer.

## Interpretation

This is exactly the mixed case: external transfer fails, but in-domain semi-real CV wins on `world` and `family` holdout.

The next issue is no longer raw representation scale alone. It is source-domain mismatch: synthetic support-contrast scores do not calibrate to semi-real score/label geometry even after adding unsafe synthetic examples.
