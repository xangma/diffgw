# NRTidalv2 Pv2 positive-22 projection receipt — 27 September 2026

**Result:** The published supplied-mode positive-frequency `(2,2)`
projection passed against an independent 80-digit scalar reference in both
Torch and JAX CPU FP64. This qualifies the operator boundary only; it is
not a native `IMRPhenomPv2_NRTidalv2` waveform gate.

The [frozen specification](../math/nrtidalv2-pv2-positive-22-projection.md)
uses [Pratten et al. (2020), Eqs. (11)–(13), (53), (93)–(94)](https://arxiv.org/html/2004.06503)
for three nondegenerate angle sets and complex supplied modes. Both complex
polarizations must agree to `1e-12` relative to maximum input-mode
magnitude, without alignment or fitting. The pre-rotation identity
`P(H+D)=P(H)+P(D)` has the same limit, and the chosen angles make direct
addition of `D` to final polarizations observably wrong. The maintained
test is `tests/test_nrtidalv2_pv2_projection_math_reference.py`.

**Execution:** host `len`, cwd
`/home/xangma/tmp/diffgw-nrtidalv2-pv2-projection-20260927`, Linux CPU
FP64, Python 3.12.3, Torch 2.14.0+cpu, JAX 0.11.2, pytest 9.1.1,
mpmath 1.3.0. Read-only hostname, disk, environment, and process checks
preceded the temporary checkout copy. The remote state-changing commands
created this temporary directory and copied the isolated checkout into it;
no package update was needed. After the final test assertion, the validation
command was:

```sh
PYTHONPATH=src:tests JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q \
  tests/test_nrtidalv2_pv2_projection_math_reference.py \
  tests/test_nrtidalv2_pv2_aligned_additive_terms_math_reference.py \
  tests/test_precession_twisting_positive_clean.py \
  tests/test_phenompv2_sourceframe_clean.py \
  tests/test_phenompv2_waveform_clean.py
```

**Final result: 36 passed, 1 skipped in 11.01 s.** The skip is the
pre-existing optional LAL/PyCBC test in the neighboring Pv2 suite.
All jobs were foreground SSH commands and completed; no PID remains.

The catalog generator's `--check` currently fails on the already-tracked
`docs/math/imrphenomxhm-projection.md`, whose four section headings do not
match the required six-section outline. This new spec has all six sections;
the unrelated page was not changed here.

The remaining block is concrete: native Pv2 has a BBH carrier/projector,
but no `IMRPhenomPv2_NRTidalv2` assembly that inserts the printed full
self-spin and signed Eq. (24) amplitude into the correct supplied mode,
then rotates it. The Eq. (24) absolute-to-mode normalization and the
published-versus-binary angle/basis contract need independent qualification.
The paper-D and historical binary-compatible variants remain separate and
untouched; no LAL source or fitted correction was used.
