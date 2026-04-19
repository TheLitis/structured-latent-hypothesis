# Context Transfer Policy v1

## Result

- `seed`: policy cost `0.476190` vs best trivial `0.380952`; does not beat best trivial baseline. Action mix `structured=0.548`, `fallback=0.095`, `escalate=0.357`.
- `world`: policy cost `0.333333` vs best trivial `0.380952`; beats best trivial baseline. Action mix `structured=0.738`, `fallback=0.143`, `escalate=0.119`.
- `family`: policy cost `0.488095` vs best trivial `0.380952`; does not beat best trivial baseline. Action mix `structured=0.833`, `fallback=0.071`, `escalate=0.095`.

## Interpretation

This probe evaluates the strongest current practical claim: whether triple-derived scores can drive an explicit accept / fallback / escalate policy with lower total cost than trivial policies.
