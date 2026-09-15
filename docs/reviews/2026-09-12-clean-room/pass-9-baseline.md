# Pass 9: PhenomD/NSBH precision-path review

This bounded review examined native PhenomD and NSBH arithmetic after the
documented phase and amplitude reviews in [pass 2](pass-2-baseline.md). It did
not revisit the unresolved calibration-table, global-gauge, remnant-selection,
or low-Lambda NSBH questions. No waveform oracle, LAL/PyCBC implementation, or
residual fit was used.

## Repair

`IMRPhenomD.compute_slope_peak` is used by the inherited NSBH default affine
gauge. Before this pass it called `as_tensor` without a dtype, which coerced
floating tensor inputs to the helper's default `float64`. In a float32 NSBH
call, the resulting batched slope term was shape `(B, 1)` and promoted the
phase to float64. A scalar slope is a 0-D operand and follows PyTorch's weak
scalar promotion rules, so the same float32 frequency grid stayed float32.
Consequently scalar and one-row NSBH calls returned `complex64` and
`complex128`, respectively, despite identical requested input precision.

The helper now promotes the floating dtypes present in its tensor arguments,
uses their device, and retains float64 for scalar-only calls. Integer tensor
inputs are converted to that selected real dtype through the same conversion
path. This changes no formula or phase/reference convention; it makes scalar
and batched calls use the same native arithmetic precision.

## Verification

The focused regression checks float32 scalar and one-row batched NSBH calls,
requiring complex64 outputs and matching polarization values with zero
absolute tolerance and a relative norm bound. It also covers mixed floating,
scalar, and integer inputs to the slope helper. The existing NSBH
batching/gradient checks and PhenomD baseline checks also pass:

```sh
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_clean_room_nsbh.py tests/test_clean_room_phenomd_baseline.py
```

Result: **21 passed**. This is a native precision consistency repair; it does
not establish strict raw-complex parity or resolve inherited fit provenance.
