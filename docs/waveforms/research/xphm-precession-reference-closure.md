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
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cpu PYTHONPATH=src pytest -q \
  tests/test_xphm_nnlo_clean_room.py \
  tests/test_xphm_rotation_reference.py \
  tests/test_clean_room_precession_equations.py \
  tests/test_clean_room_precession_conventions.py \
  tests/test_imrphenom_xphm.py \
  tests/test_pass10_nnlo_remnant.py \
  tests/test_pass12_precession.py \
  tests/parity/test_phenom_xphm_parity.py -ra
```

### NNLO-102 stacked correction

The paper-derived `frequency_domain_nnlo_geometry` route now supplies the
NNLO-102 angles on the main and per-mode SPA grids in both frontends. The
shared twist-up helper remains the sole owner of the XHM carrier adapter
$i^{m'}$; convention 1 contributes only its additional common real minus
sign. A scalar odd-$m'$ test in `tests/test_xphm_rotation_reference.py`
rejects a second application of $i^{m'}$. Published coefficient checkpoints,
stored as frozen float64 values, and Torch/JAX geometry agreement remain strict
gates in `tests/test_xphm_nnlo_clean_room.py`. That module also checks a
separate Torch-native quadrupole composition path against both production
frontends. The composition check is not an independent scalar or full-waveform
oracle, and it covers neither all modes nor all conventions.

The NNLO secular cone is unsupported where a nonaligned sample has
$L+S_\parallel\leq0$ at the reference or evaluated frequency. Such samples
produce NaN rotation angles consistently in eager Torch, compiled Torch, and
jitted JAX; exactly aligned convention-1 samples retain their gauge-fixed
reduction. A focused transform test fixes this domain behavior. These
paper/scalar gates are separate from the public-LAL binary-output measurements
in `docs/validation/xhm_xphm_binary_oracle_20260925.json`.

## 5. Reductions and failed full-waveform gates

Two native reductions are exact at the tested precision:

1. Exactly aligned, convention-1 XPHM takes the XHM carrier path.
2. XPHM restricted to `(2,2)` equals XP on their shared MSA-223/remnant-3
   route in Torch and JAX, including batched observer azimuth and selected
   azimuth derivatives. NNLO-102 instead uses the independent
   frequency-domain geometry checked in Section 4.

Those are native composition statements, not assertions that external XPHM
uses the same carrier convention. With the pinned `lalsuite` distribution
7.26.15 (LAL 7.7.1 / LALSimulation 6.2.1), the strict $10^{-6}$ complex-L2
and $10^{-8}$ mismatch gates remain red in all four fresh public-binary
diagnostics:

| Frozen case | Max polarization L2 | Max flat mismatch |
| --- | ---: | ---: |
| XHM catalog diagnostic | `0.4555338521046169` | `0.09119459041887013` |
| XPHM catalog diagnostic | `0.22881790989825165` | `0.025734053861110495` |
| XHM holdout | `0.7187377266646952` | `0.22367661976358144` |
| XPHM holdout | `0.5560211538599411` | `0.13413353052973198` |

These failures do not invalidate the scalar Wigner algebra; they leave the
higher-mode carrier and its convention as a declared, unqualified XHM
dependency. The [regional XHM carrier audit](../../math/imrphenomxhm-inspiral-amplitude.md)
locates the earliest discrepancy in higher-mode inspiral amplitude and leaves
the missing current-release calibration data explicit. Official release
documentation states that `122022` recalibrated higher-mode amplitudes without
changing phases, and the author merge request states that this recalibration
propagates to XPHM. The paper-era Appendix-A `(2,2)`/`(3,2)` mixing transform
is public, but the recalibrated spheroidal `(3,2)` amplitude inputs are not;
therefore the current release's complete mode-mixed carrier remains unclosed.

## 6. Reproduction, status, and remaining gates

The four frozen cases are generated by `tests/parity_harness.py` using only
public installed LAL binary APIs and waveform outputs. The pinned diagnostic
oracle run itself involved no source inspection, translated implementation
detail, fitted value, paper equation, or scalar-reference material. The receipt
is therefore evidence only about the measured binary outputs, separate from the
executable paper/scalar gates. This scope statement does not supersede the
historical exposure disclosure below. The manifest retains `0/1` catalog and
`0/1` holdout results for each model and the unchanged strict thresholds.

During post-implementation integration review on 2026-09-25, a previous
implementation worker inspected public LALSuite XPHM source while checking the
polarization-basis direction. That exposure occurred after the earlier
production implementation was frozen. No source excerpt, formula, translated
implementation detail, or fitted value is retained from it. The NNLO-102
correction in this stacked change was produced in a clean context from the
cited papers and independent scalar references; its external diagnostic used
only the pinned public binary interface described above. Further
production-physics work requires a clean context.

The independent `len` validation environment uses Python `3.12.3`, Torch
`2.5.0.dev20240619`, JAX/JAXlib `0.4.29`, NumPy `1.26.4`, SciPy `1.13.1`,
pytest `9.1.1`, and the pinned `lalsuite==7.26.15` distribution. It forces the
JAX CPU backend and enables FP64. The exact focused-run result is recorded in
the pull-request validation report rather than frozen into this equation
ledger.

This work does not promote XPHM, certify its XHM carrier, validate every MSA
root, or establish an NR accuracy domain. Promotion requires the relevant XHM
carrier/convention gates plus full complex-polarization catalog and holdout
passes. SpinTaylor and PNR remain separately sourced extensions with their own
open integration and calibration checks.
