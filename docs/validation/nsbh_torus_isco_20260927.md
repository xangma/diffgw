# NSBH torus and ISCO component gate — 27 September 2026

The [frozen component specification](../math/nsbh-torus-isco.md) was committed
as `f52f7ea` **before** running the native comparison. Its 105 Cartesian cases
cover $q=1,2,5,10,15$, $\chi=-0.5,0,0.5$, and
$\Lambda=0,1,10,300,800,2600,5000$. The paper's raw Appendix A Eq. (11)
fit, the Appendix A Eq. (12) Kerr ISCO radius, and the native nonnegative
carrier adapter are checked separately.

On `len`, the command `JAX_ENABLE_X64=1 PYTHONPATH=src
/home/xangma/repos/diffgw-worktrees/xas-whole-waveform-gate/.venv/bin/python
-m pytest -q -s tests/test_nsbh_torus_reference.py` ran from
`/home/xangma/tmp/diffgw-nsbh-torus-20260927` on Linux x86-64 CPU FP64.
It passed **5/5** in 13.91 s with no skips or expected failures. The focused
regression command with
`tests/test_nsbh_disruption_reference.py tests/test_nsbh_clean_amplitude.py
tests/test_clean_room_nsbh.py` passed **32/32** in 16.87 s. Logs were
`nsbh-torus-test.log` and `nsbh-regression.log` in that temporary directory.

| Measure | Torch | JAX | Frozen limit |
| --- | ---: | ---: | ---: |
| Maximum ISCO radius absolute error | $8.882\times10^{-16}$ | $8.882\times10^{-16}$ | $2\times10^{-13}$ |
| Maximum raw torus-ratio absolute error | $5.329\times10^{-15}$ | $5.329\times10^{-15}$ | $2\times10^{-12}$ |
| Maximum clipped torus-ratio absolute error | $2.776\times10^{-16}$ | $2.776\times10^{-16}$ | $2\times10^{-12}$ |

The independent scalar 70-versus-90-digit comparisons passed the fixed
$10^{-30}$ limit in all 105 cases, and both positive- and zero-torus cases
were exercised. The native derivative of $\bar r_{\rm ISCO}$ at zero spin
passed the analytic $-4\sqrt{2/3}$ slope gate in both backends.

Environment: Python 3.12.3, Torch 2.14.0+cpu, JAX 0.11.2, NumPy 2.5.3,
mpmath 1.3.0, pytest 9.1.1, Linux 6.8.0-138-generic x86-64. No LAL binary
or LAL source entered this component gate. This establishes sampled
paper-equation-to-Torch/JAX agreement, not merger-branch, full-waveform,
public-binary, or NR accuracy.
