# XPHM precession-layer reference closure

## 1. Scope and claim boundary

This ledger covers only the IMRPhenomXPHM precession-specific layer: option
routing, per-mode stationary-phase frequency routing, Wigner rotation,
positive-frequency polarization assembly, and observer-basis rotation. The
five coprecessing carriers `(2,2)`, `(2,1)`, `(3,3)`, `(3,2)`, and `(4,4)`
are a declared dependency on `IMRPhenomXHM.return_unprojected_modes`; its
fits are neither reproduced nor validated here. The remnant spin selected by
the precession option tuple is passed to that carrier through
`_final_spin_override`.

The claims below are deliberately separate:

- **paper fidelity:** the scalar rotation follows the cited equations and
  stated DiffGW carrier adapter;
- **backend fidelity:** Torch and JAX agree with that scalar calculation at
  the tested points;
- **native reduction:** selected DiffGW routes collapse to XHM or XP;
- **binary-reference compatibility:** the frozen LAL full-waveform gates
  remain failed; and
- **physical accuracy:** no new NR-domain claim is made here.

## 2. Equations, frames, and carrier adapter

The source is [Pratten et al. (2021), arXiv:2004.06503v2](https://arxiv.org/html/2004.06503v2).
Its strain convention is $h=h_+-i h_\times$ (Eq. 1), minimal rotation is
$\dot\gamma=-\dot\alpha\cos\beta$ (Eq. 7), and the inertial modes are
obtained from coprecessing modes with Wigner matrices (Eqs. 8--10). The
positive-frequency twist used here is the explicit polarization form in
Appendix E, Eqs. 110--113. Those sums use the primed spherical polarization
basis $(x'_N,y'_N)$. Appendix D, Eqs. 93--94, print the forward component map
from the unprimed LAL observer basis to that primed basis,

$$
\begin{pmatrix}h'_+\\h'_\times\end{pmatrix}
=
\begin{pmatrix}\cos 2\zeta&-\sin 2\zeta\\
\sin 2\zeta&\cos 2\zeta\end{pmatrix}
\begin{pmatrix}h_+\\h_\times\end{pmatrix}.
$$

The waveform API returns the unprimed observer basis, so the production and
scalar kernels apply the inverse map: $h_+=\cos(2\zeta)h'_+
+\sin(2\zeta)h'_\times$ and $h_\times=-\sin(2\zeta)h'_+
+\cos(2\zeta)h'_\times$. A sign-sensitive scalar test fixes both directions
at $\zeta=\pi/4$; norm preservation alone would not distinguish them.

For a positive coprecessing index $m'>0$, Appendix E Eq. 96 evaluates the
angles at the stationary-phase orbital frequency $2\pi f/m'$. Because the
native angle kernels take a dominant-harmonic frequency, XPHM passes
$f_{22}=2f/m'$. DiffGW's XHM carrier uses zero source azimuth, whereas the
paper's Appendix-C construction uses $\pi/2-\phi_\mathrm{ref}$; the explicit
adapter is therefore $i^{m'}h^L_{\ell m'}$. The remaining native
normalization is
$(-1)^\ell\sqrt{(2\ell+1)/(16\pi)}$. These adapter factors are not XHM fit
coefficients.

The independent implementation in
`src/diffgw_reference/imrphenomxphm_rotation.py` uses only the Python standard
library. It evaluates the factorial sum for $d^\ell_{m m'}(\beta)$, the
per-mode frequency map, Eqs. 110--113, and the spin-2 polarization rotation.
It imports neither production backends nor the external oracle.

## 3. Option routing contract

`src/diffgw_core/xphm_options.py` is the shared, tensor-free resolver. An
explicit `prec_version` is authoritative; the convenience
`precession_method` chooses a family only when the version is absent.

| Family | Versions | Omitted version | Omitted `final_spin_mod` | Source status |
| --- | --- | --- | --- | --- |
| NNLO | 101--104 | 102 | 0 | Pratten Appendix F, Tables 3 and 5 |
| MSA | 220--224 | 223 | 3 | Pratten Appendix F, Tables 3 and 5; public default |
| SpinTaylor | 310, 311, 320, 321 | 320 | 2 | later DiffGW numerical-precession extension |
| PNR | 300/330 (`300` normalizes to `330`) | 330 | 2 | later DiffGW PNR extension |

Convention 1 is the public default; supported conventions remain 0, 1, 5, 6,
and 7. Torch and JAX now both default to MSA version 223 with the published
MSA remnant option 3. A caller may explicitly select another supported
remnant prescription. Unsupported nonaligned MSA-remnant root configurations
still fail closed rather than silently changing the requested option.

## 4. Independent scalar and backend checks

`tests/test_xphm_rotation_reference.py` checks all five supported modes
against the scalar implementation, Wigner orthogonality for
$\ell=2,3,4$, polarization norm/composition, every $m'$ frequency map,
the option table, exact backend defaults, and version-over-method precedence.
The production helper is tested independently in both Torch and JAX; the
scalar reference does not call either one.

The wider focused suite also checks batching, selected gradients, frontend
parity, source-frame equations, and the declared reductions. The local command
is:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src pytest -q \
  tests/test_xphm_rotation_reference.py \
  tests/parity/test_phenom_xphm_parity.py \
  tests/test_clean_room_precession_equations.py \
  tests/test_clean_room_precession_conventions.py \
  tests/test_imrphenom_xphm.py \
  tests/test_pass10_nnlo_remnant.py \
  tests/test_xhm_native_fits.py -ra
```

On the local validation environment this completed with `176 passed` and
`5 skipped`; the skips require optional SXS holdout data or the `sxs` package.

## 5. Reductions and failed full-waveform gates

Two native reductions are exact at the tested precision:

1. Exactly aligned, convention-1 XPHM takes the XHM carrier path.
2. XPHM restricted to `(2,2)` equals XP for the same angle, convention, and
   remnant tuple in Torch and JAX, including batched observer azimuth and
   selected azimuth derivatives.

Those are native composition statements, not assertions that external XPHM
uses the same carrier convention. With the pinned `lalsuite` distribution
7.26.15 (LAL 7.7.1 / LALSimulation 6.2.1), the strict $10^{-6}$ complex-L2
and $10^{-8}$ mismatch gates remain red:

| Frozen case inventory rerun | XP max polarization L2 | XPHM max polarization L2 | XPHM max projection L2 |
| --- | ---: | ---: | ---: |
| catalog | `6.61e-14` | `0.22881792097112238` | `0.22867451757916643` |
| holdout | `3.60e-14` | `0.5560210336430101` | `0.6077114108352693` |

As a public-API black-box diagnostic, restricting both XPHM implementations
to LAL's mode-array `[(2,2)]` leaves plus/cross relative L2 errors of about
`0.00624` in the catalog case and `0.00581`/`0.00388` in the holdout. The
same native `(2,2)` route is exactly XP, while XP passes the frozen cases.
Thus the full failure is not evidence against the scalar Wigner algebra; it
is introduced by the XPHM carrier/mode convention beyond the native XP
reduction, with the higher-mode carrier and its convention still a declared
XHM dependency. An aligned black-box control gives max polarization L2
`0.22901036019350907` for XHM and `0.23471181102971564` for XPHM, supporting
the same diagnosis without claiming that the two external models must be
identical in that limit.

## 6. Reproduction, status, and remaining gates

The frozen cases are generated by `tests/parity_harness.py` using only public
LAL binary APIs; the implementation and reported measurements used no LAL
source or fitted value. During post-implementation integration review on
2026-09-25, the implementation worker did inspect public LALSuite XPHM source
while checking the polarization-basis direction. That exposure occurred after
the production implementation was frozen. No source excerpt, formula, or new
production-physics change is retained from it: the follow-up correction is
derived from Pratten et al. and is limited to scalar-reference naming,
documentation, and sign-sensitive tests. Any further production-physics work
requires a clean context. Mode-array diagnostics use
`SimInspiralWaveformParamsInsertModeArrayFromModeString`; LAL labels that API
unreviewed, so those numbers are diagnostic rather than a qualification gate.
The manifest retains `0/1` catalog and `0/1` holdout results and the unchanged
strict thresholds.

The independent `len` validation environment pins JAX/JAXlib `0.11.2`, NumPy
`2.5.3`, SciPy `1.18.1`, Torch `2.14.0+cpu`, and pytest `9.1.1`; it forces the
JAX CPU backend and enables FP64. The focused run completed in 435.71 seconds
with `175 passed` and `7 skipped`. The skips were the unavailable optional
LAL/PyCBC smoke test, SXS data/package tests, and a historical-artifact check
that needs Git history; the latter passed in the local checkout.

This work does not promote XPHM, certify its XHM carrier, validate every MSA
root, or establish an NR accuracy domain. Promotion requires the relevant XHM
carrier/convention gates plus full complex-polarization catalog and holdout
passes. SpinTaylor and PNR remain separately sourced extensions with their own
open integration and calibration checks.
