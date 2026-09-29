# NSBH branch amplitude factors gate — 27 September 2026

The [frozen specification](../math/nsbh-branch-amplitude-fits.md) was
committed as `3f0d8f1` **before** native comparison. It transcribes
[Thompson et al. (2020)](https://arxiv.org/pdf/2002.08383v1), Appendix A
Eqs. (20)–(24), (30), and the three dimensionless factors in Table I.
The half factor in Eq. (30) is explicit. The scalar reference uses paper
decimals and high-precision arithmetic; no LAL source or binary waveform
serves as a component oracle.

The frozen gate covers 30 controlled cases at $\Lambda=0,0.5,1,2,800$,
including all four classes and $f_{\rm tide}=f_{\rm RD}$, plus 180 cases
composed from the existing independent disruption and torus references.
All 210 cases passed the 70-versus-90-digit scalar stability limit
$10^{-30}$. Torch and JAX CPU FP64 passed every frozen native comparison:

| Maximum error | Torch | JAX | Frozen limit |
| --- | ---: | ---: | ---: |
| Adjusted ringdown frequency, relative | $2.237\times10^{-16}$ | $2.237\times10^{-16}$ | $2\times10^{-13}$ |
| Fit coordinates and branch factors, absolute | $3.331\times10^{-16}$ | $3.331\times10^{-16}$ | $2\times10^{-12}$ |

On `len`, the command `JAX_ENABLE_X64=1 PYTHONPATH=src
/home/xangma/repos/diffgw-worktrees/xas-whole-waveform-gate/.venv/bin/python
-m pytest -q -s tests/test_nsbh_branch_amplitude_fits.py
tests/test_nsbh_branch_reference.py tests/test_nsbh_torus_reference.py
tests/test_nsbh_disruption_reference.py tests/test_nsbh_clean_amplitude.py
tests/test_clean_room_nsbh.py tests/parity/test_nsbh_parity.py` ran from
`/home/xangma/tmp/diffgw-nsbh-branch-fits-20260927` on Linux x86-64 CPU
FP64. The log is `pytest.log` in that directory.
The combined fit and neighboring NSBH run passed **101/101** in 74.30 s,
with no skips or expected failures.
The test was subsequently renamed to
`tests/test_nsbh_branch_amplitude_fits_math_reference.py` for the strict
CI gate; the command above records the name used for this run.

Environment: Python 3.12.3, Torch 2.14.0+cpu, JAX 0.11.2, NumPy 2.5.3,
mpmath 1.3.0, pytest 9.1.1. This qualifies the branch factors at supplied
$f_{\rm RD}$; it does not qualify the remnant/QNM provider, Table I joining
frequencies, Eq. (26)/(29) corrections, amplitude windows, assembled
waveform, or physical calibration.
