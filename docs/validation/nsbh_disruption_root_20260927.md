# NSBH disruption-root component gate — 27 September 2026

The [frozen component specification](../math/nsbh-disruption-root.md) was
committed as `2fe41a4` before any comparison. It fixes 105 compactness/root
cases, the 60-case positive-denominator radius/frequency subset, and all
thresholds. The source is [Thompson et al. (2020), Appendix A Eqs. (8)–(10)
and Appendix B Eq. (32)](https://arxiv.org/pdf/2002.08383v1). The independent
`mpmath` reference brackets the rightmost root of the transformed polynomial;
NumPy companion-root enumeration independently checks root selection.

The run used host `len`, cwd
`/home/xangma/tmp/diffgw-nsbh-disruption-20260927`, Linux CPU FP64,
Python from `/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python`,
Torch 2.14.0+cpu, JAX 0.11.2, mpmath 1.3.0, SciPy 1.18.1, and pytest 9.1.1.
The command was `JAX_ENABLE_X64=1 PYTHONPATH=src python -m pytest -q -s
tests/test_nsbh_disruption_reference.py` using that Python. It passed 5/5
tests in 14.82 s. The related `test_clean_room_nsbh.py`,
`test_pass18_nsbh_classification.py`, and `parity/test_nsbh_parity.py`
suite passed 76/76 tests in 45.47 s on the same host and environment.

| Backend | Max absolute $C$ error | Max relative $\xi$ error | Max normalized polynomial residual | Max relative radius error | Max relative frequency error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Torch | 2.220e-16 | 7.430e-16 | 1.397e-16 | 7.408e-16 | 1.069e-15 |
| JAX | 2.220e-16 | 8.025e-16 | 1.146e-16 | 7.408e-16 | 9.897e-16 |

All errors are below the precommitted limits of `2e-13` absolute for
compactness, `2e-11` relative for the root, `2e-12` for normalized residual,
and `2e-10` relative for radius/frequency. The root check also recomputes the
polynomial independently of the backend-returned residual.

No public LALSuite binary was used in this gate. Its waveform output does
not expose the disruption root or frequency, and the other NSBH components
are still provisional. This receipt certifies only the specified paper-math
slice, not an assembled waveform or NR calibration.
