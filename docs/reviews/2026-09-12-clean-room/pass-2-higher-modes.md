# XHM published phase connections and mode mixing

Date: 2026-09-12. This continues the bounded [first HM/XHM audit](higher-modes.md).
It establishes the equations and numerical properties listed here, not complete
IMRPhenomXHM calibration or LAL agreement. No oracle calls, implementation
source, translated ports, old review material, or residual fitting were used.
The coordinator's report of a failed compatibility check prompted the independent
phase-gauge review below; no oracle data were used to derive or tune a change.
No fitted coefficients were changed.

The C1 inspiral reconstruction is now available only through
`IMRPhenomXHM(experimental_phase_connections=True)`. The default retains the
inherited carrier-based inspiral and value-only phase join. Its inspiral join
is therefore not claimed to be C1. The matrix conversion, mixed ringdown
boundaries, exact derivatives, and numerical normalization remain active.

## Fresh primary evidence

The primary source is [García-Quirós et al., arXiv:2001.10914v2](https://arxiv.org/pdf/2001.10914v2),
dated 3 November 2020. Its PDF SHA-256 is
`00f3bdbe28a6ccde5cde3af82ac8fc95153ec6f03a4b071ce0ed57326e4e09ee`.
The [version history](https://arxiv.org/abs/2001.10914) still identifies v2 as the
latest version. The attached [mathematical-asset README](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/README)
does not resolve the fit-target and collocation ambiguities below. No resolving
erratum was found in the fresh search.

Only mathematical `.m` expression/statistics files from the author ancillary
archive were downloaded. Generated C and implementation notebooks were not
downloaded or consulted. The asset inventory and hashes follow below. Matching
an inherited expression to a mathematical asset does not certify how its output
must be assembled into a waveform.

## Implemented equations

- **PN phase through linear frequency.** Equation (4.9) asks for the argument of
  the complex PN amplitude re-expanded through linear `Mf`, rather than its
  unexpanded argument. Appendix E, Eqs. (E10)-(E12), (E14), gives the
  frequency-dependent term `Lambda_lm = c_lm (2 pi/m) Mf`, after removing the
  constant complex prefactor. The coefficients are
  `c_21 = -(1+ln 16)/2`, `c_33 = -21/5 + 6 ln(3/2)`,
  `c_32 = (3-66 eta/5)/(1-3 eta)`, and
  `c_44 = [eta(3579-2880 ln 2)-1008+960 ln 2]/[120(3 eta-1)]`.
  The odd-mode imaginary coefficients vanish at exact equal mass at the PN
  order used, so that branch takes zero. Constant arguments and the large-mass-
  ratio fitted branch are separate unresolved conventions below.
- **Experimental C1 inspiral phase.** Equation (4.11) is assembled as
  `phi_lm = (m/2) phi_22(2 Mf/m) + Lambda_lm + dphiIns_lm Mf + phiIns_lm`.
  At the inspiral cut, choose `dphiIns_lm` from the intermediate derivative
  minus the scaled carrier derivative and `Lambda_lm'`. The inherited constant
  mode convention is retained, and the intermediate constant is solved from
  phase continuity. Nonmixing ringdown retains its existing C1 construction.
  This enforces the published connections, without certifying the intermediate
  calibration values or absolute phase/time convention. Both the linear PN
  term and this connection are opt-in because their sum is fixed entirely by
  the currently unverified intermediate derivative gauge.
- **Spherical 32 reconstruction.** Appendix A, Eq. (A3), gives the two-mode
  matrix relation `h_Y = alpha h_S`. The available quadrupole is spherical.
  Eliminating the first spheroidal unknown gives
  `h_32,Y = (alpha_322/alpha_222) h_22,Y +
  [alpha_332-(alpha_322/alpha_222) alpha_232] h_32,S`.
  This replaces direct addition of a spherical quadrupole as if it were
  spheroidal. The generic helper takes the matrix entries in published row
  order; no conjugation is inserted into Eq. (A3). Mapping the inherited fit
  names to these entries remains an explicitly qualified dependency below.
- **Mixed ringdown boundaries.** Construct the spherical complex ringdown
  before solving the intermediate regions. Its magnitude and derivative supply
  the amplitude boundary at `f_RD_amp = f_ring_22`. Its phase derivative and
  second derivative supply the two ringdown constraints of the 32 intermediate
  phase at `f_RD_phase = f_ring_22 - f_damp_22/2`, as required in Sec. V B.
  Both regions therefore connect to the same spherical waveform. The previous
  final hard switch between an unmixed and mixed complete strain was removed.
  The remaining constant phase shift is applied to the complete spherical mode.
- **Exact local derivatives and normalization.** Frequency derivatives use
  autograd with parameter graphs retained, replacing finite frequency steps.
  Carrier probes use geometric distance `D=M`, and the shared leading SPA
  envelope is divided out before mixing. This avoids taking high derivatives of
  tiny physical strain in float32. The physical distance factor and mode
  projection normalization are restored once in the existing output path.
  Factoring out the spheroidal phase avoids wrapping the entire rapidly varying
  carrier phase into a principal argument. Internal probes also work under
  `torch.no_grad` and `torch.inference_mode`; inference outputs remain detached.

## Phase-gauge review and default isolation

Section V explicitly states that the intermediate phase calibration determines
the relative time shifts among modes. Its role is not merely interpolation
between otherwise anchored inspiral and ringdown pieces. Equation (4.11) is
therefore safe to assemble only once the intermediate derivative-fit targets
and their time convention are known.

Write the scaled carrier phase as `P(f)` and the computed intermediate phase
derivative as `I'(f)`. The implemented PN expansion has `Lambda(f)=c f`.
The C1 condition gives `dphiIns = I'(f_ins)-P'(f_ins)-c`; hence the total
inspiral correction is exactly `[I'(f_ins)-P'(f_ins)] f`. The PN coefficient
cancels. Compared with the inherited value-only connection, the complete phase
change is
`Delta phi(f) = [I'(f_ins)-P'(f_ins)] min(f, f_ins)`.
The post-inspiral constant follows from continuity. This holds for the 32
mode as well once its mixed ringdown and intermediate reconstruction are fixed.

Thus the first pass-2 default imported an unverified calibrated time gauge into
the inspiral. This is an identified reconstruction dependency, not a claim that
the reported binary discrepancy has been decomposed by mode. Inferring a
conversion for the author fits from a binary discrepancy would be calibration
from residuals and was not done. The experimental option preserves the
published connection equations and their tests while keeping that unsupported
gauge assignment out of the default. It does not certify the inherited default
or resolve the intermediate/ringdown calibration gaps below.

## Validation

Run from `/Users/xangma/repos/torchwave`:

```sh
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q tests/test_clean_higher_modes.py tests/test_phenomhm_reconstruction.py tests/test_phenomxhm_reconstruction.py tests/test_imrphenom_hm.py
```

Result after phase-gauge isolation: **55 passed in 7.49 s**.
`git diff --check` passed.

The added pass-2 cases independently expand the complex PN coefficients,
check the two simultaneous matrix equations and complex gradient checks, compare
complete-waveform one-sided phase derivatives at every higher-mode phase cut,
and check the 32 amplitude cuts and C2 ringdown phase connection. All C1
assertions remain exercised by the experimental path; default-path cases check
phase values at the inspiral cut and derivatives at the other cuts. A synthetic
intermediate-fit offset leaves every default inspiral mode unchanged and changes
the experimental inspiral as expected from the derivation above. Batched float32
and float64 inference evaluations agree with ordinary no-grad evaluations in
both paths.
The first-pass covariance, equal-mass, body-exchange, and full-model gradient
checks remain in the suite. These are equation and regularity checks, not
calibration or oracle comparisons. No existing tolerance was relaxed.

## Unresolved reconstruction dependencies

1. **Intermediate fit targets and nodes.** The author 21 `p1`, `p2`, and `p4`
   expressions produce much smaller values than `p3` and `p5`, consistent with
   different or relative targets, but the inspected mathematical expressions
   do not define their conversion to physical phase derivatives. No conversion
   was invented. The inherited assembly treats them as direct derivatives.
   Figure 8's default subset and Table VII also need reconciliation with the
   inherited choice of the first five nodes. The 32 intermediate fit assembly
   has the same unverified target-convention dependency.
2. **32 spheroidal ringdown fits.** Table IX specifies four collocation
   frequencies and Eq. (6.6) has four unknown coefficients. The attached archive
   instead exposes `dphiS_p1`, `dphiS_fring32`, and `dphiS_p4`, including a middle
   frequency absent from that table. The inherited three-condition solve with
   `alpha2_32 = alpha2_22` is retained and explicitly marked uncertified. The
   carrier-alignment frequencies used with `deltaPhiS` and `deltadPhiS` also
   remain unverified. Neither smooth connections nor exact derivatives supply
   these missing calibration definitions.
3. **Mixing coefficient convention.** The algebraic helper is independently
   established. The inherited `mixing_alpha223` and `mixing_alpha333` names are
   mapped to the matrix entries `alpha_232` and `alpha_332`, respectively.
   Three real-part mathematical assets were compared, but the complete complex
   index/sign convention is not independently certified. The published matrix
   identity test deliberately uses an independently chosen general matrix.
4. **Absolute mode phase and fitted PN branch.** Inherited constant offsets,
   including the 21 sign decision at `Mf=0.008`, are retained without new
   certification. Their relation to Eq. (4.13), the carrier reference gauge,
   and spherical/spheroidal phase alignment remains unresolved. Equation
   (4.10)'s fitted phase branch at high mass ratio has not been assembled from
   the author fits because its calibrated derivative convention is unresolved.
5. **Amplitude and other first-pass gaps.** This pass does not certify the
   fitted amplitude normalizations, unexpanded 21 branch, collocation vetoes,
   interval polynomial fallback logic, all QNM fits, projection conventions,
   or the HM PN/constant-gauge dependencies listed in the first audit.

These are dependencies on missing or unresolved primary mathematical metadata,
not evidence that the currently inherited choices agree with a target binary.

## Mathematical asset inventory

All paths below are relative to
`https://arxiv.org/src/2001.10914v2/anc/`; the local downloads are in
`/tmp/torchwave-clean-higher-modes/anc/`.

| Asset | SHA-256 |
| --- | --- |
| [MixingCoefficients/CoefficientStatsre_alpha222.m](https://arxiv.org/src/2001.10914v2/anc/MixingCoefficients/CoefficientStatsre_alpha222.m) | `eb7708484832f2ca3a086b51bcc59f4fd2514c2e52633f26aa676c26a45ab781` |
| [MixingCoefficients/CoefficientStatsre_alpha322.m](https://arxiv.org/src/2001.10914v2/anc/MixingCoefficients/CoefficientStatsre_alpha322.m) | `1162a9205e597481938f29c0a76c857e3b17c0fec02de2a46471c95d3df3ab24` |
| [MixingCoefficients/CoefficientStatsre_alpha333.m](https://arxiv.org/src/2001.10914v2/anc/MixingCoefficients/CoefficientStatsre_alpha333.m) | `c16b67ba0b66b5f6f0cf48d035d0a28fac2af211dc554fe1348d13cc3c964334` |
| [ParspaceFits/PhaseFits/21/3Dfit_p1_lm_21.m](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/21/3Dfit_p1_lm_21.m) | `428c1f23c4d1ca5e2c5935d4f0f0ce7c7e11cc45be79e64f4636064df5440935` |
| [ParspaceFits/PhaseFits/21/3Dfit_p2_lm_21.m](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/21/3Dfit_p2_lm_21.m) | `a3ac2ef4b308abd7f2de64d05e45020c3e97f10160da9efd9c888c78b213bca0` |
| [ParspaceFits/PhaseFits/21/3Dfit_p3_lm_21.m](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/21/3Dfit_p3_lm_21.m) | `60acb8aa7a6dcf1b25896d4fb2b2f51a03c19e3b86837868eb44fc30dd6f386e` |
| [ParspaceFits/PhaseFits/21/3Dfit_p4_lm_21.m](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/21/3Dfit_p4_lm_21.m) | `9d350c7eae43ff62fce1193fb2d4e62c30c07b00f431ea6798ba40245b3063d7` |
| [ParspaceFits/PhaseFits/21/3Dfit_p5_lm_21.m](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/21/3Dfit_p5_lm_21.m) | `b2c63176675569bed679bcd5c98e7cc91f825752985b070b047c26f1c79bd2f8` |
| [ParspaceFits/PhaseFits/32/deltaPhiS.m](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/32/deltaPhiS.m) | `e92a6657458934bea6bb8466b97c7cd90a45d57d98eff9b546e7d64804fac772` |
| [ParspaceFits/PhaseFits/32/deltadPhiS.m](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/32/deltadPhiS.m) | `07107d0436b316792e0b2cd343b8cce3efa2a96276a26304f6ecbdfb7c975455` |
| [ParspaceFits/PhaseFits/32/dphiS_fring32.m](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/32/dphiS_fring32.m) | `1a1b3519df07ffef8a0a065184a3ce66417895569ba540852877b8f7c4ca3ddf` |
| [ParspaceFits/PhaseFits/32/dphiS_p1.m](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/32/dphiS_p1.m) | `3c6ad2109a6d85202f794579c88b9b52ebe9b39340e1aa9f02ef4fa1f01b89d7` |
| [ParspaceFits/PhaseFits/32/dphiS_p4.m](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/32/dphiS_p4.m) | `c417a56b22950dae88c782f35d4470b22af92073cffe92dc273e8e71c155d082` |
