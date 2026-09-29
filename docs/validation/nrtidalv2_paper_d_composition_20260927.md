# NRTidalv2 spinning paper-D composition receipt — 27 September 2026

**Result:** The frozen positive-Lambda, nonzero-spin complex quotient gate
passed for Torch and JAX CPU FP64 across the printed premerger-to-taper
boundary. It is conditional on the printed equations and the existing
IMRPhenomD carrier. The full waveform is not thereby qualified.

The [specification](../math/nrtidalv2-paper-d-composition.md) freezes three
cases, 12 frequencies per case, both polarizations, the scalar
phase/self-spin/taper composition, and a `1e-10` absolute complex quotient
limit without phase/time alignment or amplitude fitting. The printed-fit
merger frequencies are about 2102.760106 Hz for the equal case and
1673.800358 Hz for both unequal labelings. The strictly post-endpoint
quotient is exactly zero. The test is
`tests/test_nrtidalv2_paper_d_composition_math_reference.py`.

**Execution:** host `len`, cwd
`/home/xangma/tmp/diffgw-nrtidalv2-paper-d-composition-20260927`,
Linux CPU FP64, Python 3.12.3, Torch 2.14.0+cpu, JAX 0.11.2,
pytest 9.1.1, mpmath 1.3.0. Command:

```sh
PYTHONPATH=src:tests JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q \
  tests/test_nrtidalv2_paper_d_composition_math_reference.py \
  tests/test_nrtidalv2_self_spin_math_reference.py \
  tests/test_nrtidalv2_phase_math_reference.py \
  tests/test_nrtidalv2_amplitude_math_reference_native.py \
  tests/test_nrtidal.py
```

Result: **75 passed, 1 skipped in 22.08 s**. The skip is the existing
optional LALSimulation/PyCBC comparison in `tests/test_nrtidal.py`.
Output is `paper_d_composition.log` under that cwd. The foreground command
completed in 24.4 s including SSH; no process remains.

No pinned-LAL binary gate was added: its D option does not have the exact
paper Table II amplitude contract, and its nonzero-tide effective merger
boundary differs from the printed fit. The paper's Fig. 7 example also
conflicts with Eqs. (28)–(29). Those are source/binary disagreements rather
than failures of this conditional printed-equation composition gate; see
the [self-spin receipt](nrtidalv2_self_spin_20260927.md) and
[merger provenance audit](nrtidalv2_provenance_20260927.md). A spinning
paper-D binary or independent physical/NR gate remains open.
