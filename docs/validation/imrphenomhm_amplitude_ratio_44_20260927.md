# IMRPhenomHM leading (4,4) amplitude-ratio gate — 27 September 2026

The [source-first v1 specification](../math/imrphenomhm-amplitude-ratio-44.md)
was frozen in commit `87d1898` before the independent reference and tests.
[London et al., arXiv:1708.00404v4, Eqs. (3)–(4)](https://arxiv.org/pdf/1708.00404v4)
and [Mishra et al., arXiv:1601.05588v1, Eqs. (11), (12a), (12f)](https://arxiv.org/pdf/1601.05588v1)
fix the restricted leading (4,4) amplitude ratio. The scalar source reduces
the full SPA-amplitude quotient symbolically, evaluates it with mpmath, and
composes the independent Eq. (5) map. The public gate removes the public
PhenomD carrier magnitude from the public HM unprojected (4,4) magnitude.

Host `len`, cwd `/home/xangma/tmp/diffgw-hm44-source-gate-20260927`;
source copied from isolated branch `codex/hm-leading-44-amplitude` based on
`b111a76`, with reference/test changes recorded in `d4021cf`. Python:
`/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python`. Packages:
Python 3.12, mpmath 1.3.0, SymPy 1.14.0, NumPy 2.5.3, Torch 2.14.0+cpu,
JAX 0.11.2, pytest 9.1.1, and PyCBC 2.11.0 from a separate compatible
package directory. The exact final command was:

```sh
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
PYTHONPATH=/home/xangma/tmp/diffgw-pycbc-2.11.0-full/packages:src:tests \
/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q \
tests/test_imrphenomhm_amplitude_ratio_math_reference.py \
--reference-device=cpu -ra
```

Result: **15 passed in 14.23 s**, no skips or warnings. Log:
`/home/xangma/tmp/diffgw-hm44-source-gate-20260927/hm44-pytest-rerun.log`.
The initial run had 14 passes and one test-fixture failure: synthetic
`f_{22}^{RD}=0.05` made the map's middle and upper slopes identical for
`m=4`, contradicting the test's asserted kink. Correcting only that
synthetic fixture to 0.045 produced the passing result. The frozen
equation, tolerance, and test grid were unchanged.

This qualifies the sampled leading (4,4) rescaling component through the
public D-carrier assembly. It does not independently qualify the D carrier,
QNM inputs, (4,4) phase or projection, higher PN amplitude terms, other
modes, physical accuracy, or the complete HM waveform.
