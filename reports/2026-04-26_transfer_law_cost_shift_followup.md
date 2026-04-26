# Transfer Law Cost-Shift Follow-Up

## Main Result

The policy law was evaluated under `24` unseen cost-ratio settings after selecting thresholds with source costs from the previous policy sweep.

`0` source policies beat the best trivial baseline on every target setting and every `seed/world/family` split.

## Interpretation

This closes the strongest cost-invariant reading of the policy result. The law is not robust to arbitrary unseen cost ratios in the current benchmark.

The surviving claim is narrower: triple-derived scores can support a useful transfer decision policy **after cost calibration**, but they do not yet provide a cost-model-free router.

## Next Step

The next strict test is a semi-real transition benchmark or a family of context shifts that changes visual style while preserving the same transfer-decision labels.
