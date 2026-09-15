# Published precession conventions and wrapper limits

This pass implements Pratten Table IV in XP/XPHM and checks the associated
observer geometry. It changes only the precession wrappers, narrow raw-angle
plumbing in the existing NNLO/MSA evaluators, and focused tests. It does not
change precession-version defaults, PN coefficients, calibration, remnant fits,
or the legacy PhenomP frame construction. No external waveform implementation,
source port, binary oracle, or historical calibration report was used.

## Primary sources

Files were obtained from the primary authors/publishers and kept outside the
repository in `/tmp/torchwave-clean-room-precession`. SHA-256 identifies the
exact mathematical references:

| Source | Equations used | SHA-256 |
|---|---|---|
| [Pratten et al., 2004.06503v2](https://arxiv.org/pdf/2004.06503v2) | Table IV; 3.1, 3.5–3.7, 4.4; C4–C24; D6–D7; E18–E19 | `fd21fcd7c111d780eddb109e51c5813d7645a52022b84a24d923056b639bd935` |
| [Bohé et al., T1500602-v4](https://dcc.ligo.org/public/0122/T1500602/004/PhenomPv2_technicalnotes.pdf) | Mathematical frame derivation 37–40; opening-angle discussion 22–24 | `5847dd72a26b25eb75d3913c474e65cd0e6025a1637f95424b9ed01cfd033f02` |
| [Schmidt et al., 1703.01076v1](https://arxiv.org/pdf/1703.01076v1) | Source-to-observer rotation, Eq. 31 | `49f7e7f7acac8bd2971fdd8c81572a9159d5aaa7b9b089d20891e9eebaab03b2` |
| [Chatziioannou et al., 1703.03967v3 source](https://arxiv.org/src/1703.03967v3) | Raw MSA primitives and integration constants, Eq. 66 and Appendix F | `64f6114885f1c6d5ab5afdaaa0d6cf44d65eb3a97f7a5b84df324397a39fdf92` |

The technical note and papers discuss software interfaces as well as equations.
Automatically rendered paper text included interface names and a generic
surrogate-interface example. No waveform implementation or linked code was
inspected or used; the derivation uses the equations listed above.

## Table IV implementation

Write `a(f), e(f)` for the raw analytical primitives, with `gamma=-epsilon`.
The resulting angles and carrier phase are:

| Convention | alpha | epsilon | Carrier orbital phase |
|---|---|---|---|
| 0 | a(f)-a(ref)+alpha0_XP | e(f)-e(ref) | phi_JL0 |
| 1 | a(f)-a(ref)+alpha0_Pv2 | e(f)-e(ref)+epsilon0_XP | 0 |
| 5 | a(f)+alpha0_Pv2 | e(f) | input phiRef |
| 6 | a(f)-a(ref)+alpha0_XP | e(f)-e(ref)+epsilon0_XP | input phiRef |
| 7 | a(f)+alpha0_XP | e(f) | input phiRef |

Raw access is opt-in (`anchor_reference=False`); existing callers retain
reference subtraction. `evaluate_msa_precession` retains its five-result API
and optionally supplies the carrier phase as a sixth result. Both wrappers
apply that phase to their native carrier. XPHM evaluates the angle frequency
for each coprecessing m separately, retaining the common reference frequency.

All conventions use the actual angle between J and N and the C22 polarization
basis rotation. The output transformation is the inverse of D6–D7:
`hp=cos(2*zeta)*hp_raw+sin(2*zeta)*hc_raw`,
`hc=-sin(2*zeta)*hp_raw+cos(2*zeta)*hc_raw`.
The previously derived source-azimuth factor `i**m` remains unchanged.

Pv2 alpha is computed independently using the observer-to-J Cartesian
rotation in T1500602 Eqs. 37–40. Algebraically expanding its x component avoids
subtracting nearly equal Jz-squared terms for small cones. Quadrant and
small-cone tests compare it with a direct Cartesian basis construction.

## Observer azimuth and batching

`phi0` denotes an additional source-frame observer azimuth. C4 and C24 use
`phiRef-phi0` inside N and the observer polarization vector P, while Table IV's
carrier argument remains the original `phiRef`. This rotates N and P by
positive `phi0` about the source z axis. In the aligned limit the published
spin-weighted harmonics give the factor `exp(-i*m*phi0)` in both polarizations.
XP's aligned XAS call therefore uses `phi_c=phic-phi0`; XPHM passes its observer
azimuth to XHM. The analytical precessing branches use the same geometry.
The experimental ODE branch rejects nonzero observer azimuth when a precessing
rotation is required, since its observer-frame construction is incomplete.

Reference frequency now participates in parameter broadcasting, including
when it is the only batched input. XPHM's aligned blend passes the original
frequency grid to XHM, preserving scalar/mixed-batch equivalence.

## J opposite to L: explicit unsupported secular domain

The signed geometric opening angle in Pratten Eq. 4.4 approaches pi when
`S_perp -> 0` and `L+S_parallel<0`. It must not be collapsed to zero. However,
that geometric correction alone cannot fix the waveform's aligned limit:
the printed NNLO coefficients satisfy `alpha_dyn=epsilon_dyn` as chi_p tends
to zero. The surviving Wigner phase is `exp[i*m*(alpha-epsilon)]` at beta=0
but `exp[-i*m*(alpha+epsilon)]` at beta=pi. The latter retains a nonconstant
secular phase. Constant Table IV offsets cannot remove it.

T1500602 Eq. 24 explicitly chooses a positive cosine to preserve the aligned
limit during flips, while Pratten Eq. 4.4 gives the signed cosine. Pratten
Sec. III excludes transitional precession from the fixed-J approximation.
These sources do not supply a consistent signed-cone NNLO phase repair.

The implementation consequently rejects NNLO and actual NNLO-fallback rows
requiring a precessing rotation when `L+S_parallel<=0` at the reference or any
evaluated angle frequency. It does not reject valid MSA rows merely because
their cone exceeds pi/2. This is an explicit conservative support boundary,
not a claim that a physical binary is undefined. Exactly aligned convention-1
waveforms remain the separate physical nonprecessing carrier, including mixed
batches. Their near-aligned NNLO neighbors can reject in this unsupported
domain; no continuity claim is made across that boundary. Nondefault exact
anti-J inputs reject because their raw secular twist would be required. The
low-level geometric beta retains its pi branch.

## Source sufficiency and unresolved choices

1. With the same J and the physical source adapter of Schmidt Eq. 31, the
   published Pv2 and XP alpha constructions coincide modulo 2pi. Table IV
   names them separately but does not provide an additional legacy adapter
   that makes them distinct. This implementation does not infer such an
   adapter from external outputs.
2. The Pv2 note illustrates a 2PN nonspinning J; Pratten Table III separately
   selects the angular momentum by version. Table IV does not unambiguously
   prescribe a second 2PN J solely for alpha0_Pv2. The implementation uses
   one J with the selected version's angular momentum, consistently with C5.
3. Unanchored MSA primitives have free additive integration constants. Options
   5/7 use the printed primitives with zero added constants. The primary
   papers do not settle the gauge an external implementation might choose
   internally for those raw primitives.
4. Options 0/5/7 retain spin-azimuth gauge dependence as transverse spin tends
   to zero. At exactly zero transverse J the declared gauge is phi_JL0=0.
   Default 1 and option 6 have the supported aligned limit from every azimuth.
5. Previously recorded MSA version distinctions, oscillatory phase branch
   limitations, carrier calibration, precessed-remnant omissions, and the
   experimental ODE frame limitations remain outside this narrow pass.

## Independent verification

`tests/test_clean_room_precession_conventions.py` checks every table row for
NNLO 102 and MSA 223 against raw primitives and independent Cartesian geometry;
carrier phase identities for all supported modes; reference subtraction;
quadrants and tiny cones; scalar/batch equivalence and finite-difference
gradients; nonzero observer azimuth and aligned harmonic factors; XP/XPHM22
agreement; default/nondefault domain errors; mixed aligned-row preservation;
reference and per-mode domain checks; and valid MSA cones above pi/2.

The combined precession test command excludes the two external-oracle tests:

```text
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest tests/test_clean_room_precession_conventions.py tests/test_clean_room_precession_equations.py tests/test_clean_room_msa.py tests/test_imrphenomp.py tests/test_imrphenom_xphm.py tests/test_precession_ode.py tests/test_phenomxp_msa_reconstruction.py -k 'not lal' -q
```

Result: **107 passed, 2 external-oracle tests deselected** in 10.76 seconds;
the focused convention file contains 51 passing tests. `git diff --check`
also passed. These equation and invariant checks do not certify external
binary parity.

The [independent review](pass-4-precession-conventions-independent-review.md)
records the resolved findings, separate regression probes and reviewed hashes.
Final whole-repository verification passed 745 tests with one CUDA-only skip;
the strict catalog and holdouts still fail at 11/21 and 4/13 passing cases.
See [pass-5 evidence](verification/pass-5/) and the [current report](summary.md).
