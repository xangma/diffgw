# IMRPhenomT dominant ringdown supplied-input gate, 2026-09-27

The [component specification](../math/phenomt-ringdown-frequency.md)
records the printed sign conflict between Estellés et al. THM Eqs. (25)
and (27a). The tested branch uses the minus sign inferred from Eq. (27a)
and the peak join in Eq. (31b). This is a conditional equation-level
qualification, not a claim that the literal Eq. (25) or the calibrated
input fits are verified.

On `len` CPU, a clean snapshot of integration commit `3ac8cc3` plus this
slice was tested in
`/home/xangma/tmp/diffgw-phenomt-ringdown-20260927`.
Python `/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python` was 3.12.3,
with NumPy 2.5.3, PyTorch 2.14.0+cpu, JAX 0.11.2, mpmath 1.3.0, and
pytest 9.1.1. The current installed CPU packages were used; no environment
update was needed.

```sh
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src:tests /home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q tests/test_phenomt_ringdown_frequency_math_reference.py --reference-device=cpu -ra
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src:tests /home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q tests/test_phenomt_ringdown_frequency_math_reference.py tests/test_phenomt_reference.py::test_author_reference_high_spin_unequal_masses tests/test_phenomt_remnant.py::test_remnant_override_preserves_merger_and_ringdown_join --reference-device=cpu -ra
```

The isolated gate returned **7 passed in 11.12 s**. The combined gate
returned **11 passed in 16.83 s**, with no skips. Three binary states
`(30,30,0,0)`, `(35,25,0.7,-0.5)`, and `(60,6,0.9,0.1)` supplied the
native peak/QNM/shape inputs. At six nonnegative times, both frontends'
production carrier frequencies, phase increments, and analytic slopes
agreed with the 80-digit scalar formulas at the specification's FP64
tolerances. The scalar identities also passed independent quadrature and
finite-difference checks. The existing pinned author regression and
remnant-override join checks passed in the combined run; the override
equations themselves remain outside this component's source-backed gate.

No native waveform code, constants, defaults, or calibration were changed.
The input fits, amplitudes, other modes, full-family assembly, and physical
accuracy still need separate evidence.
