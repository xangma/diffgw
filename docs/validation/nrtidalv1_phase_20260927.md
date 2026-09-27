# NRTidal v1 nonspinning tidal-phase qualification — 27 September 2026

**Outcome: 33/33 passed**, no skips, expected failures or deselection, for
the [bounded phase specification](../math/nrtidalv1-phase.md). This does not
qualify the complete waveform or its physical accuracy.

The run used `len`, Linux 6.8.0-138-generic x86_64, in
`/home/xangma/tmp/diffgw-nrtidalv1-phase-20260927`, copied from isolated
checkout `518e49a33c6da8cdae860959c81ddc985cd3f512` with this slice's
working files. The command was:

```sh
DIFFGW_REQUIRE_LAL_BINARY=1 OMP_NUM_THREADS=1 \
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
PYTHONPATH=/home/xangma/tmp/diffgw-nrtidalv1-phase-20260927/src:/home/xangma/tmp/diffgw-nrtidalv1-phase-20260927/tests \
/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q -s \
  tests/test_nrtidalv1_phase_math_reference.py \
  tests/test_nrtidalv1_phase_lal_binary.py
```

The complete log is `qualification.log` in that directory. Worker PID
`1485390` exited after 17.78 s. Python was 3.12.3; NumPy 2.5.3,
SymPy 1.14.0, mpmath 1.3.0, Torch 2.14.0+cpu, JAX 0.11.2, and LALSuite
7.26.15. The installed public LALSimulation extension was
`_lalsimulation.cpython-312-x86_64-linux-gnu.so`, SHA-256
`e816879d23e652e8b3928496d6fd261de0832b5c3781c50e33a458042495b42e`.
No LAL source was inspected.

The scalar reference derives the effective tidal coupling from Dietrich et
al. (2017) Eq. (2) and transcribes its Eq. (7) Padé phase. It symbolically
checks the mass-weighted deformability and exact 1PN coefficient; native
Torch/JAX unit-complex BNS/BBH quotients match the explicitly rounded-carrier
scalar values at five frequencies between 30 and 800 Hz. Selected frequency
derivatives at 300 Hz meet the specified $10^{-8}$ relative and
$10^{-10}$ rad/Hz absolute limits.

The predeclared binary comparison used the four main systems plus two
holdouts stated in the specification, all at 30–800 Hz in 5 Hz steps. It
compared *unaligned* unit-complex `IMRPhenomD_NRTidal / IMRPhenomD`
quotients, requiring relative complex $L_2\le10^{-6}$ and maximum phase
residual $\le10^{-5}$ rad. The largest observed LAL-to-rounded-scalar
relative complex $L_2$ was `2.09995954e-15`; the largest Torch/JAX-to-LAL
value was `2.66248986e-15`. The largest respective pointwise phase
residuals were `9.01201102e-15` and `1.2543367e-14` rad. These results
show the native implementations, independent scalar rounded variant, and
named binary agree on the declared phase quotient.

**Source-to-carrier precision boundary.** Eq. (7) defines
$d_1=n_1-3115/1248=-19.923993589\ldots$, whereas the native carriers
and named binary use the rounded `-19.924`. This changes the paper-literal
phase by up to `1.6497434e-5` rad for the equal-mass case and
`1.9064569e-5` rad for the unequal-mass case over the same grid. The
corresponding paper-literal versus rounded unit-complex relative $L_2$
values are `5.7321178e-6` and `6.6094838e-6`. The binary match therefore
validates a three-decimal realization of the published fit, not exact
implementation of its analytic 1PN constraint.

Amplitude, merger/taper, spinning binaries, other modes, FP32 and accelerator
behavior, inherited IMRPhenomD physics, and comparisons with independent
neutron-star simulations remain outside this receipt.
