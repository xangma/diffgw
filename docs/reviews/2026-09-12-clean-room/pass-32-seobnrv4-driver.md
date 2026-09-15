# Pass 32: SEOBNRv4 start-frequency safety and unresolved driver choices

Date: 2026-09-13

This pass advanced the clean-room SEOBNRv4 driver without selecting an
unsupported calibration convention.  The implementation was checked against
the primary SEOBNRv4 paper ([Bohé et al., arXiv:1611.03703](https://arxiv.org/abs/1611.03703))
and the public TorchWave API contract.  No LALSimulation, PyCBC, or TEOBResumS
implementation source was inspected.

## Concrete milestone

Both `SEOBNRv4Published` and the native `SEOBNR_ODE` path now reject an
explicit `f_lower` unless it is finite and strictly positive.  The check is
performed before quasi-circular initial data or trajectory construction, with
the same error contract in both paths:

```text
ValueError: f_lower must be finite and positive
```

This follows the documented meaning of `f_lower` as the positive starting GW
frequency.  The driver converts it to the dimensionless orbital frequency

\[
M\Omega=\pi M f_{\rm lower},
\]

so zero, negative, NaN, and infinite values cannot define a physical initial
state.  When `f_lower` is omitted, the existing behavior remains: the lowest
positive sample in the requested grid is used, and an all-zero grid requires
an explicit lower bound.

## Why no unresolved physics choice was changed

The highest-impact unresolved driver choice remains the spinning-flux
truncation.  The primary equations expose both the finite tabulated
spin-dependent mode content and the general leading-\(\ell=8\) expression;
the paper does not state which one is the intended SEOBNRv4 implementation
branch.  The other source-supported blockers remain:

* the Bohé v6 residual-phase energy convention is not uniquely fixed by the
  paper's notation and surrounding equations;
* the SEOBNRv4 paper invokes the Hofmann final-spin formula without choosing
  one of its published coefficient blocks;
* the peak-event refinement and interpolation stencil are required by the
  matching equations but are not specified as a unique numerical algorithm;
* the raw NR calibration trajectories are not supplied as a machine-readable
  table in the primary paper, so recovering them from public waveform output
  would be oracle fitting.

Consequently this pass changes no phase, flux, remnant-fit, event, or
attachment convention and makes no parity claim.

## Validation

Focused command:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q \
  tests/test_clean_room_eob_published_driver.py \
  tests/test_clean_room_eob_waveform.py
```

Result: **35 passed**.  The new tests cover zero, negative, NaN, and infinite
explicit start frequencies and assert that rejection happens before trajectory
initialization/reconstruction.
