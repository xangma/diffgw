# TEOB native fixed-event reassembly

This diagnostic reconstructs the cached trajectories from
`2026-09-15-no-contact/teob-ode-convergence-clean/convergence-arrays.npz`.
For every cached `(time, state)` series it evaluates the native clean TEOB
vector field to recover orbital frequency, then runs the native waveform,
NQC, and ringdown assemblers. The public mode is used only for scoring; no
saved correction factor or oracle mode is used to construct output.

Run from the repository root:

```sh
PYTHONPATH=src /Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
  scripts/teob_fixed_event_replay.py \
  --output docs/reviews/2026-09-15-parity-strategy-experiment/eob/native-fixed-event-reassembly.json
PYTHONPATH=src /Users/xangma/miniconda3/envs/pycbcorig313/bin/pytest -q \
  tests/test_teob_fixed_event_replay.py
```

## Results

The bounded local run completed in about 4 seconds on the local host. Native
baseline output reproduces the saved processed public-grid array to relative
L2 `2.43e-16`; its independent score against the public mode is `1.763e-3`.
The native report records NQC amplitude/phase matrices, coefficients,
condition numbers, matching indices, event indices, and signed-amplitude
branch counts.

The factorial controls hold the ringdown peak time fixed at the baseline
native value (`12316.531403647654 M`) while varying trajectory level and NQC
auxiliary spacing (`0.05`, `0.1`, `0.2 M`). The baseline output scores are
`1.803e-3`, `1.763e-3`, and `7.329e-3`; the tight levels and all matching
indices are in the JSON. The event probe holds trajectory and NQC inputs fixed
and shifts only the native peak by `-0.05`, `0`, `+0.05 M`, producing scores
`2.192e-3`, `1.763e-3`, and `1.495e-3`. State perturbations of ±`1e-8` give
native-output amplification about `1.84`; no corrected-amplitude branch
crossing occurred.

The earlier `fixed-event-replay.json` cache-ratio experiment is withdrawn: it
filled the tail with the public target and could not diagnose native NQC.
Its saved-array pair check remains useful context: tight_0p1→tight_0p01 had
pre-NQC relative L2 `2.676e-8`, processed relative L2 `2.165e-3`, and
attachment shift `-0.02737 M`.

## Controlled 2x2 replay

The requested pair is `tight_0p1` versus `tight_0p01`. A 200-knot grid from
`12298.65990319574` through `12318.559903195739 M` lies within both native
spline supports and is passed to both reassemblies. The diagnostic also uses
a local clone made from `inspect.getsource` with asserted replacements for
the auxiliary grid and NQC event times; package code is unchanged. The clone
matches native output exactly (relative L2 `0` for both levels).

Pairwise output relative L2 values are:

| Controls | Relative L2 |
| --- | ---: |
| Native controls | `2.1647e-3` |
| Common auxiliary grid | `1.4726e-7` |
| Fixed NQC and ringdown events | `7.9988e-4` |
| Common grid plus fixed events | `9.0398e-7` |

The common-grid rows have identical auxiliary times. In the final factorial
rows, both NQC event times (`12316.487269626203 M` and
`12315.487269626203 M`) and the ringdown peak (`12316.531451506382 M`) are
bitwise identical across levels. Matching indices are recorded per row since
fixed physical times can land on different knots on different grids. These
controls strongly localize the observed `~2.16e-3` inter-trajectory spread
to the native auxiliary-grid/support choice; they do not establish a package
fix or eliminate all native interpolation effects.

The event probe is a local wrapper around the native peak-polynomial helper;
its `-0.05`, `0`, `+0.05 M` shifts are relative to the native fitted
ringdown peak, and zero goes through the same override path. Cached states
avoid ODE reintegration, but the cache has no serialized frequency, so
frequency reconstruction assumes the same clean Hamiltonian and
radiation-reaction policy as the generator.
