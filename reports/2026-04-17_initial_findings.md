# Initial Findings - 2026-04-17

## Run Metadata

- Worlds: `commutative`, `noncommutative`
- Variants: `baseline`, `smooth`, `comm`, `comm_step`
- Seeds: `3, 11, 29`
- Script: `scripts/run_synthetic.py`

## Main Result

The current benchmark does **not** yet support the practical claim that CFP improves held-out composition performance.

## Quantitative Snapshot

### Commutative World

- `baseline` test reconstruction MSE: `0.000906`
- `comm` test reconstruction MSE: `0.001110`
- `comm_step` test reconstruction MSE: `0.002106`
- `baseline` commutativity error: `0.002042`
- `comm` commutativity error: `0.000674`
- `comm_step` commutativity error: `0.000107`

Interpretation: CFP is successfully enforcing flatter latent geometry, but on this first setup that geometric pressure is not translating into better held-out reconstruction.

### Non-Commutative World

- `baseline` test reconstruction MSE: `0.002136`
- `comm` test reconstruction MSE: `0.002340`
- `comm_step` test reconstruction MSE: `0.003183`
- `baseline` commutativity error: `0.017538`
- `comm` commutativity error: `0.017400`
- `comm_step` commutativity error: `0.003146`

Interpretation: the stronger CFP variant reduces commutativity error even in the control, but this comes with worse task performance. That is consistent with over-regularization rather than clean structure capture.

## What This Means

The first run gives a useful negative-but-informative result:

1. The regularizer is numerically active and measurable.
2. The benchmark can distinguish geometry pressure from task quality.
3. The current lambda values or split design are not yet producing the desired commutative-world gain.

## Immediate Next Moves

1. Sweep `lambda_comm` downward and add a softer warmup.
2. Reduce the strength of `comm_step`; it is currently too destructive.
3. Run at least two latent sizes to see whether CFP helps under tighter capacity.
4. Tighten the benchmark so the held-out split stresses composition more than plain reconstruction.
