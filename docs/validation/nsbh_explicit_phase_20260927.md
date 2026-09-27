# NSBH explicit-gauge phase composition — 27 September 2026

The [v1 specification](../math/nsbh-explicit-phase.md) was frozen in
commit `2f68cff` before scalar/native comparison. Thompson et al.
[Sec. II.2](https://arxiv.org/html/2002.08383v2) prescribe a PhenomD
phase plus the updated NRTidal phase. The scalar expression composes the
independent PhenomD and NRTidalv2 references; it does not import the
Torch/JAX NSBH implementations or LAL code.

On `len`, Linux x86-64 CPU FP64, in
`/home/xangma/tmp/diffgw-nsbh-explicit-phase-20260927`, the command
`PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/home/xangma/repos/diffgw-worktrees/xas-whole-waveform-gate/.venv/bin/python
-m pytest -q tests/test_nsbh_explicit_phase_math_reference.py` passed
**10/10** in 13.83 s with no skips or expected failures. Log:
`nsbh-phase-pytest.log`. The frozen sample has mass ratios 5 and 9,
zero and nonzero NS deformability, four frequency points spanning the
PhenomD phase regions, and both explicit phase gauges. The scalar 70/90
digit check passed the `1e-30` absolute phase limit; both native backends
passed the `3e-7` maximum unit-phasor error limit. No numerical alignment
or binary-derived fit was applied.

The neighboring command using the same environment,
`python -m pytest -q tests/test_imrphenomd_waveform_math_reference.py
tests/test_nrtidalv2_phase_math_reference.py tests/test_nsbh_clean_phase.py`,
passed **189/189** in 386.81 s, with no skips or expected failures. Log:
`nsbh-phase-neighbors.log`. Environment: Python 3.12.3, Torch 2.14.0+cpu,
JAX 0.11.2, SymPy 1.14.0, mpmath 1.3.0, pytest 9.1.1.

The gate conditions on the native-supplied final spin/QNM pair and the
declared PhenomD fit rows with digits beyond the printed Khan table. It
therefore qualifies the explicit phase **composition at those inputs**, not
the remnant/QNM fit or its coefficient provenance. The legacy grid-dependent
phase offsets, disruption-dependent amplitude, conflicting published/native
amplitude variants, complex whole waveform, and NR validation remain open.
