# NRTidalv2 Pv2 aligned-carrier receipt — 27 September 2026

**Result:** The conditional, pre-rotation BNS carrier assembled from a
supplied face-on BBH carrier and the published finite-size terms passed in
Torch and JAX CPU FP64. This is not a complete precessing BNS waveform gate.

The [frozen specification](../math/nrtidalv2-pv2-aligned-carrier.md) follows
[Dietrich et al. (2019), Eqs. (3)–(4), (17)–(29), Table II and Sec.
III.4](https://arxiv.org/html/1905.06011). The signed Eq. (24) amplitude
is added to the BBH amplitude; the printed tidal plus 2PN, 3PN and 3.5PN
self-spin phases then act on that sum. The merger-to-`1.2 f_mrg` Planck
window is applied before rotation. The scalar reference composes the
existing independent 80-digit SymPy/mpmath phase, spin, amplitude and
taper references. The native Torch/JAX function accepts a supplied BBH
carrier and computes every finite-size term without calling that reference.

Three spinning positive-Lambda systems, including exchanged bodies, span
100 Hz, 800 Hz and eight fractions of the printed merger frequency. Native
IMRPhenomD supplies the face-on BBH carrier to both the native composition
and scalar reference, so this is conditional on that separately validated
carrier. At supplied nonzero angles the complete assembled carrier also
passes through the native positive-22 projector against the independent
scalar projector. Both complex carriers and polarizations meet the frozen
`1e-10` normalized limit without phase/time alignment, refitting or output
rescaling; post-taper bins are exactly zero. No LAL source or binary residual
was used.

**Execution:** host `len`, cwd
`/home/xangma/tmp/diffgw-nrtidalv2-pv2-carrier-20260927`, Linux CPU FP64;
Python environment
`/home/xangma/repos/diffgw-worktrees/xas-whole-waveform-gate/.venv/bin/python`
with Torch 2.14.0+cpu, JAX 0.11.2, mpmath 1.3.0, SymPy 1.14.0 and
pytest 9.1.1. Read-only environment and process diagnostics preceded
copying the isolated checkout into owned scratch. Other active jobs were
left alone. The scoped command was:

```sh
PYTHONPATH=src:tests JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
/home/xangma/repos/diffgw-worktrees/xas-whole-waveform-gate/.venv/bin/python \
  -m pytest -q tests/test_nrtidalv2_pv2_aligned_carrier_math_reference.py
```

**Result: 6 passed in 10.15 s.** A combined run of the new gate and the
existing aligned-additive, tidal-mode, projection, paper-D and Pv2 driver
tests returned **52 passed, 1 skipped in 18.20 s**. The skip is the
existing optional LAL/PyCBC check. Both SSH test processes exited; no
background job or PID remains to stop.

The helper is not wired to the public Pv2 BNS dispatch. The published
insertion order is source-complete, while BNS precession angles, remnant and
spline-time conventions, and reference-frequency anchoring remain to be
fixed before claiming whole-waveform parity. The Eq. (28)/Fig. 7 Love–Q
and printed merger-fit/pinned-binary conflicts remain open. The historical
`binary_compatible` default was not changed.
