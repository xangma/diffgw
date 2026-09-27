# HM outer-map selection gate — 27 September 2026

The [six-section specification](../math/imrphenomhm-frequency-map.md)
freezes the [London et al. (2018) Eq. (5)/(8) cutoffs](https://arxiv.org/html/1708.00404v4):
`0.018 R/Q` for amplitude and `0.014 R/Q` for phase. The new public
`london_2018` option selects these outer-map joins for the D carrier; the
default `phenomd_runtime` retains the established reverse pair. The scalar
reference receives QNM frequencies as inputs and imports neither frontend.

On `len`, CPU FP64, the integrated code at `833ba03` passed **67/67** in
17.44 s for `tests/test_imrphenomhm_frequency_map_math_reference.py` and
`tests/test_imrphenomhm_phase_offsets_math_reference.py`, using
`JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src:tests OMP_NUM_THREADS=1`
and `--reference-device=cpu -ra`. The gate checks public `(3,3)` Torch/JAX
class and dispatcher calls, map values and one-sided slopes at both joins,
and identical complex mode values when the prior default is selected
explicitly. The later integrated NSBH/HM focused run passed **142/142**.

This evidence qualifies the named outer-map selection component at the
sampled points. It does not qualify QNM fits, carrier joins, phase offsets,
amplitude calibration, projection, or the whole HM waveform.
