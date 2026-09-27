# NRTidalv2 Pv2 tidal-mode normalization receipt — 27 September 2026

**Result:** The published Eq. (24) finite-size amplitude, converted from
face-on plus strain to the supplied positive-frequency `(2,-2)` mode and
projected at supplied angles, passed the independent scalar gate in Torch
and JAX CPU FP64. This is a conditional component result, not qualification
of a complete precessing BNS waveform.

The [frozen specification](../math/nrtidalv2-pv2-tidal-mode-normalization.md)
derives the factor `2/sqrt(5/(4 pi))` from the zero-angle positive-mode
projection in [Pratten et al. (2020), Eqs. (11)–(13)](https://arxiv.org/html/2004.06503).
The signed Eq. (24) term follows [Dietrich et al. (2019), Eqs. (22)–(25)](https://arxiv.org/html/1905.06011),
whose Eq. (22) substitutes luminosity for the effective distance of
[Hotokezaka et al. (2016), Eqs. (15), (17), (24)](https://arxiv.org/html/1603.01286).
That cited definition includes inclination and detector response in effective
distance and therefore fixes the inherited coefficient as face-on plus
strain. The 80-digit scalar mode conversion and projection reference has
no native Torch/JAX or LAL import.

Three spinning, positive-Lambda systems, two angle sets, three live
frequencies, and an exactly zero post-taper bin are fixed in the maintained
test. Both complex polarizations and the supplied mode must match to
`1e-10` pointwise relative to the nonzero tidal-amplitude/taper scale. The
zero-angle inclination law is a separate closed-form check. No alignment,
fitting, output rescaling, or LAL code was used.

**Execution:** host `len`, cwd
`/home/xangma/tmp/diffgw-nrtidalv2-pv2-mode-20260927`, Linux CPU FP64,
Python 3.12.3, Torch 2.14.0+cpu, JAX 0.11.2, mpmath 1.3.0, SymPy 1.14.0,
pytest 9.1.1. Read-only environment, process, and directory diagnostics
preceded copying this isolated checkout into owned scratch. The root
integration job observed on len was left alone. No package update was
needed. The validation command was:

```sh
PYTHONPATH=src:tests JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
/home/xangma/repos/diffgw-worktrees/xas-whole-waveform-gate/.venv/bin/python -m pytest -q \
  tests/test_nrtidalv2_pv2_tidal_mode_normalization.py \
  tests/test_nrtidalv2_pv2_projection_math_reference.py \
  tests/test_nrtidalv2_pv2_aligned_additive_terms_math_reference.py
```

**Result: 15 passed in 14.03 s.** The regression command in the same
environment ran `tests/test_phenompv2_waveform_clean.py` and
`tests/test_phenompv2_sourceframe_clean.py`: **24 passed, 1 skipped in
3.88 s**. After a test-loop cleanup and documentation update, a final
combined run of all five test modules returned **39 passed, 1 skipped in
13.88 s**. The skip is the existing optional LAL/PyCBC test. All background
SSH jobs completed; PIDs `343874`, `360215`, and `417858` no longer exist.
Their logs are `pytest.log`, `pytest-pv2-regression.log`, and
`pytest-final.log` in the scratch cwd.

The remaining gate is native Pv2 BNS assembly: a source-correct aligned
BBH carrier with the finite-size phase and amplitude inserted before
precession, plus the BNS angle and reference-frequency convention. This
receipt tests supplied angles and an additive tidal mode, not that whole
waveform. The Love–Q/Fig. 7 and printed merger-frequency/binary conflicts
remain open. Historical `binary_compatible` D behavior is unchanged.
