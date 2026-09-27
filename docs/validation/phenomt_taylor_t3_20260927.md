# IMRPhenomT TaylorT3 component receipt — 2026-09-27

`T-PN-T3-v1` covers the aligned `(2,2)` 3.5PN TaylorT3 coefficient table,
frequency polynomial, and phase primitive. It does not cover pseudo-PN fits,
calibrated `t0`, joins, modes beyond `(2,2)`, or full waveforms.

## Source and implementation

The independent scalar equations are transcribed from Estellés et al.,
[T/TP Appendix A, Eqs. (A5a)–(A5h)](https://arxiv.org/pdf/2004.08302v2),
and the updated [THM Eqs. (8)–(11)](https://arxiv.org/pdf/2012.11923v1).
The printed A5g omits `235925 eta³/1769472`, which the pinned MIT
[phenomxpy author source](https://gitlab.com/imrphenom-dev/phenomxpy/-/blob/4121fafa0484f3db2a7352f4eea697faaaf29f65/phenomxpy/phenomt/internals.py)
explicitly identifies and includes. The reference exposes `paper_literal`
and `author_carrier`; native Torch/JAX use the latter. See the
[six-section specification](../math/phenomt-taylor-t3.md).

The scalar reference uses mpmath exact rationals and the paper's compact
mass-fraction spin combinations. Native Torch/JAX use expanded tensor
polynomials; they do not import the reference. The pre-existing collocation
test now takes its six PN inputs from this scalar reference rather than from
the implementation under test.

## Reproduction and result

Host: `len` (CPU). Isolated cwd:
`/home/xangma/tmp/diffgw-phenomt-t3-20260927`.
Command:

```sh
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src:tests \
  /home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q \
  tests/test_phenomt_taylor_t3_math_reference.py \
  tests/test_phenomt_inspiral_collocation_math_reference.py \
  --reference-device=cpu -ra
```

Result: **22 passed in 11.26 s**, zero skipped/xfail, exit 0. Log:
`/home/xangma/tmp/diffgw-phenomt-t3-20260927/phenomt-t3-pytest.log`.
Versions: Python 3.12.3, NumPy 2.5.3, PyTorch 2.14.0+cpu, JAX 0.11.2,
mpmath 1.3.0, pytest 9.1.1. The strict repository-wide
`--require-math-reference` switch requires collection of every math-reference
module and is reserved for the full integration suite.

Five physical mass/spin cases include equal masses and spins, unequal spins,
`q=10`, and negative/positive anti-aligned combinations. Each backend's six
coefficients agree with the scalar values within `rtol=2e-12, atol=2e-13`;
frequency at `theta={0.33,0.55,0.82}` within the same limits; and phase
within `rtol=2e-12, atol=2e-12`. The scalar derivative recovers frequency
to relative `1e-35` at 80 digits. The paper/carrier 3PN difference is checked
as an exact-rational mpmath identity. These thresholds bound float64
evaluation roundoff, not model error against numerical relativity or LAL.
