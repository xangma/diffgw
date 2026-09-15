# Pass 31: IMRPhenomNSBH reference sentinel review

## Scope

This pass audits the `IMRPhenomNSBH` reference-frequency sentinel and tensor
behavior without reading or translating LALSuite, PyCBC, or TEOBResumS source.
The evidence is the published NSBH/PhenomD API contract, TorchWave's own
implementation, and public waveform calls used only as black boxes.

## Finding and fix

The public waveform API accepts `f_ref=0` as the lower-band reference
sentinel. A black-box call with a 20 Hz lower band agrees with an explicit
20 Hz reference (the maximum complex-bin difference was exactly `0.0` in the
public binary check). The same public call rejects `f_ref=-1` with an
input-domain error. TorchWave previously treated every nonpositive value as the zero
sentinel, so negative references were accepted incorrectly.

The implementation now rejects negative and nonfinite `f_ref` values, while
zero still resolves to the lowest positive frequency in the supplied grid.
For a custom grid containing no positive samples, the waveform amplitude is
identically zero; the phase carrier uses the documented 30 Hz omitted-reference
default to keep the result finite rather than introducing an arbitrary 1 Hz
fallback. This case is outside the generated positive-frequency waveform
domain and is covered only as a finite-output tensor behavior.

## Tensor checks

The focused tests cover a leading DC/negative bin, all-nonpositive grids,
batched parameters on a shared frequency grid, float32 behavior from the
existing suite, and finite gradients through the sentinel path. A batched
zero sentinel agrees with scalar calls using the shared grid's 20 Hz lowest
positive frequency.

## Verification

```
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_clean_room_nsbh.py tests/test_nsbh.py \
  tests/test_phenomnsbh_reconstruction.py
36 passed in 1.00s
```

`git diff --check` passes. No oracle residual was used to choose a constant,
and no LAL/PyCBC/TEOBResumS implementation source was inspected or copied.
