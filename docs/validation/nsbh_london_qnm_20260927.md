# NSBH London 220 QNM input — 27 September 2026

The [v1 specification](../math/nsbh-london-qnm.md) was frozen in commit
`fedcae7` before native comparison. Its five complex polynomial terms come
from [London and Fauchon-Jones, Eq. (22)](https://arxiv.org/pdf/1810.03550v2),
with the frequency and quality-factor conversions given by
[Thompson et al., Appendix A Eqs. (16)–(19)](https://arxiv.org/html/2002.08383v2).
The independent scalar reference evaluates only those printed coefficients;
it imports no native NSBH helper or LAL implementation.

On `len`, Linux x86-64 CPU FP64, the combined command
`PYTHONPATH=src:tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
JAX_ENABLE_X64=1 /home/xangma/.cache/diffgw-xp-pycbc-venv/bin/python
-m pytest -q tests/test_nsbh.py tests/test_clean_room_nsbh.py
tests/test_nsbh_phase_mass_regression.py
tests/test_nsbh_explicit_phase_math_reference.py
tests/test_nsbh_branch_reference.py
tests/test_nsbh_london_qnm_math_reference.py` passed **92/92** in 23.02 s,
without skips or expected failures. Working directory:
`/home/xangma/tmp/diffgw-nsbh-london-qnm-20260927`; log:
`nsbh-london-combined.log`. Environment: Python 3.12.3, Torch 2.14.0+cpu,
JAX 0.11.2, mpmath 1.3.0, pytest 9.1.1, with PyCBC installed for the
strict LAL binary neighbor tests.

The 18 supplied remnant states passed the frozen 70/90-digit scalar
stability limit `1e-50`; both native backends passed the relative
`3e-12` frequency and quality-factor limits, positivity, and inverse-mass
scaling. The first neighboring run in a Python environment without PyCBC
had 22 strict binary cases fail because their required `optimized_match`
was unavailable; all 22 passed unchanged in the PyCBC-capable environment
above. No gate or tolerance was relaxed.

This certifies implementation of the published London 220 fit at supplied
final mass and spin. It does not qualify the remnant provider, the distinct
PhenomD phase QNM interpolation, amplitude assembly, or whole waveform.
Zappa and Jiménez-Forteza remnant tables print fewer coefficient digits
than the native fits, so a published-precision interval gate remains a
separate prerequisite.
