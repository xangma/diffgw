# IMRPhenomHM leading (3,3) amplitude-ratio gate — 27 September 2026

The [source-first v1 specification](../math/imrphenomhm-amplitude-ratio.md)
was frozen in commit `7249360` before the independent reference and tests.
[London et al. Eqs. (3)–(4)](https://arxiv.org/html/1708.00404v4) and
[Mishra et al. Eqs. (11), (12a), (12c)](https://arxiv.org/html/1601.05588v1)
fix the leading restricted stationary-phase `(3,3)` amplitude ratio. The
scalar reference reduces their full leading SPA amplitudes with SymPy, then
uses mpmath to evaluate the positive-mass-ordering ratio and its one-sided
frequency slopes. The test divides the public unprojected HM `(3,3)` mode
magnitude by the public PhenomD `(2,2)` carrier magnitude at the mapped
frequency. It exercises both the historical and opt-in London map cutoffs.

On `len`, from the integrated checkout archived at
`/home/xangma/tmp/diffgw-integration-gate-20260927h` (local commit
`1a49502`), CPU FP64 Torch/JAX tests passed **94/94** in 31.39 s. The command
used `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1`,
`PYTHONPATH=/home/xangma/tmp/diffgw-pycbc-2.11.0-full/packages:src:tests`,
and the existing len venv's Python to run `pytest -q` over
`test_imrphenomhm_amplitude_ratio_math_reference.py`,
`test_imrphenomhm_frequency_map_math_reference.py`,
`test_imrphenomhm_phase_offsets_math_reference.py`, `test_imrphenom_hm.py`,
and `test_phenomhm_reconstruction.py` with `--reference-device=cpu -ra`.
PyCBC 2.11.0 came from a separate compatible package directory. There were
no skips or failures; one pre-existing Torch JIT deprecation warning remains.

This qualifies the leading positive-`delta` `(3,3)` rescaling component at
the sampled D-carrier boundary. Other mode ratios, complex spin terms,
equal-mass zeros and sign conventions, the carrier and QNM fits, phase,
projection, and complete HM waveform are separate gates.
