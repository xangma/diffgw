# NRTidalv2 Pv2 aligned-frame additive-term receipt — 27 September 2026

**Result:** The frozen aligned-frame, signed additive tidal term passed for
Torch and JAX CPU FP64 across the printed merger/taper interval. The gate
does not qualify a complete precessing waveform.

The [frozen specification](../math/nrtidalv2-pv2-aligned-additive-terms.md)
defines the Eq. (24) contribution with the printed 2PN, 3PN and 3.5PN
self-spin phase. It fixes three spinning positive-Lambda systems, ten
frequencies per system, and a `1e-10` relative complex limit with no
alignment or fitting. The additional 3.5PN term is demonstrably nontrivial
at 800 Hz in the frozen equal-mass case. The strictly post-taper term is
exactly zero in both frontends. The maintained test is
`tests/test_nrtidalv2_pv2_aligned_additive_terms_math_reference.py`.

**Execution:** host `len`, cwd
`/home/xangma/tmp/diffgw-nrtidalv2-pv2-additive-20260927`, Linux CPU FP64,
Python 3.12.3, Torch 2.14.0+cpu, JAX 0.11.2, pytest 9.1.1 and mpmath
1.3.0. Read-only host/environment/disk diagnostics preceded the temporary
checkout copy. The validation command was:

```sh
PYTHONPATH=src:tests JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q \
  tests/test_nrtidalv2_pv2_aligned_additive_terms_math_reference.py \
  tests/test_nrtidalv2_paper_d_composition_math_reference.py \
  tests/test_nrtidalv2_self_spin_math_reference.py \
  tests/test_nrtidalv2_phase_math_reference.py \
  tests/test_nrtidalv2_amplitude_math_reference_native.py \
  tests/test_nrtidal_reference.py tests/test_nrtidal.py
```

Result: **91 passed, 1 skipped in 25.19 s**. The skip is the pre-existing
optional LALSimulation/PyCBC check in `tests/test_nrtidal.py`. The foreground
SSH command completed in 27.5 s; no job remains. The earlier narrower run
passed 62 tests in 17.48 s. No package update was needed.

No pinned-LAL binary gate was added: this checkout has no native
`IMRPhenomPv2_NRTidalv2` assembly to compare under an exact binary
convention. The paper places the corrections before precession rotation,
so applying the Eq. (24) absolute amplitude directly to final plus/cross
polarizations would not be this published contract. The printed
Love–Q/Fig. 7 and merger-frequency/binary discrepancies remain as recorded
in the prior [self-spin](nrtidalv2_self_spin_20260927.md) and
[provenance](nrtidalv2_provenance_20260927.md) receipts. Historical
`binary_compatible` behavior was untouched.
