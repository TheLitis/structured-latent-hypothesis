# Audited Cross-Family Summary - 2026-04-18

## Families

- `ramp_v2`: mostly negative after the anchor fix
- `scale_v2`: strongest audited support
- `rotation_v1`: mixed but non-trivially positive in part of the ladder

## Main Takeaway

After auditing the benchmark contract, the evidence for CFP is weaker than the pre-audit story, but cleaner.

The current honest statement is:

> CFP is not a universal win, and it is not just generic smoothness. The best audited evidence is a bounded advantage region in the scale family, with a weaker and less stable echo in the rotation family.

## What This Means

- The old "exactly commuting point" narrative is no longer supported by the audited ladders. Baseline wins at the shared `alpha=0` anchor in all three families.
- The strongest current signal is more subtle: CFP can outperform baseline at some non-zero commutator values without `smooth` matching it.
- Because the winning worlds are family-dependent and not monotone in the commutator scalar, the next stage must test specificity more carefully before any comparison with external popular baselines.

## Next Step

Run the audited `scale_v2` and `rotation_v1` ladders again with:

1. `step_only` as the missing internal ablation;
2. nested lambda selection instead of manual `0.010/0.050`;
3. only then move to comparisons against broader standard methods.
