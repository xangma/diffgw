# Bounded independent HM/XHM reconstruction audit

Date: 2026-09-12. Starting checkout: `84ef9b3`. This record covers only the
HM/XHM changes listed below. Passing these checks does not certify complete
IMRPhenomHM or IMRPhenomXHM reconstruction or agreement with a waveform oracle.

The later [pass-2 audit](pass-2-higher-modes.md) records additional XHM phase
connections and spherical 32 reconstruction. The unresolved-dependency list in
this first-pass record describes the state before those subsequent changes.

## Sources and boundary

The primary papers were downloaded afresh. Full PDFs, extracted text, and the
one ancillary mathematical asset examined are outside the repository in
`/tmp/torchwave-clean-higher-modes/`. No oracle waveform calls, implementation
source, translated oracle ports, git history, or prior review/calibration material
were used for these changes. The inherited TorchWave source was treated as
uncertified input. An inherited reference to an old provenance ledger was
replaced with this bounded record; that ledger was not inspected.

| Primary source | Version and SHA-256 | Mathematical use |
| --- | --- | --- |
| [London et al., Mapping the dominant gravitational wave harmonic](https://arxiv.org/pdf/1708.00404v4) | v4; `079521aebaccec86c4079901c7f5e56b2ed4712cf79320327d903f46d565a8d7` | Eqs. (3)-(9), especially transition values in the text after Eq. (5) and the alternative amplitude ratio after Eq. (4). |
| [García-Quirós et al., IMRPhenomXHM](https://arxiv.org/pdf/2001.10914v2) | v2; `00f3bdbe28a6ccde5cde3af82ac8fc95153ec6f03a4b071ce0ed57326e4e09ee` | Eq. (E8), leading terms in Eqs. (E10)-(E12), (E14); Eq. (2.13), time-shift discussion in Sec. II B; Secs. IV-VI for unresolved components. |
| [Author mathematical fit statistics, AmpFits/21/CoefficientStatsInt1.m](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/AmpFits/21/CoefficientStatsInt1.m) | attached to v2; `e0c25f0119cffdf50552a4e9d78a92470f0e4c28e57f83ed94aad7d5d1427fdf` | Only the nonspinning intermediate-amplitude expression was independently compared with the inherited `amp_int1_21` expression. This limited match does not validate the other fits, their normalization, or the waveform reconstruction. |

The author ancillary directory and README were inspected to identify the
mathematical asset. Generated C and notebook implementation sections were not
used. No coefficient values were changed or fitted in this work.

## Changes and derivation

1. **HM transition frequencies.** Set geometric amplitude transition to
   `0.018 * f_RD_lm / f_RD_22` and phase transition to
   `0.014 * f_RD_lm / f_RD_22`, as stated after London Eq. (5). The inherited
   forward path had these values reversed. One existing derivative test also
   contained the reversed phase value; its input was corrected with its original
   accuracy threshold retained.

2. **HM SPA envelope.** Use the alternative amplitude prescription explicitly
   stated after London Eq. (4): `beta_lm = H_lm(f) / H_22(f_map_A)`.
   TorchWave's `h_hat_multipole` removes the common Fourier SPA envelope.
   Restoring its ratio multiplies the inherited dimensionless multipole by
   `[(2 f / m) / f_map_A]^(-7/6)`. This factor is one during the inspiral
   mapping and generally differs from one above it. This implements the paper's
   stated alternative; it does not claim to reproduce every implementation's
   choice between that alternative and Eq. (4)'s default prescription.
   This choice follows the inherited numerator, which already evaluates the PN
   multipole at the requested mode frequency `f`; it restores that construction's
   omitted denominator envelope without adding different mapped-frequency PN
   factors. Whether this is the intended target implementation's prescription
   remains an unresolved compatibility question. It is not asserted to be
   LAL-equivalent.

3. **XHM SPA normalization.** Eq. (E8) gives an envelope proportional to
   `v_m^(-7/2)`, where `v_m = (2 pi f / m)^(1/3)`. Dividing by the quadrupole
   envelope at the same frequency gives `(m/2)^(7/6)`. The inherited code used
   its reciprocal. Independent leading-PN tests cover 21, 33, 32, and 44.

4. **XHM extrinsic covariance.** Evaluate the carrier with zero coalescence time
   and orbital phase throughout intrinsic mode construction. Apply
   `exp[-i(2 pi f tc - m phic)]` once to each completed mode, including 32
   mixing. Previously the phase joins turned a time shift into a constant above
   the inspiral join for some modes. The new placement obeys the Fourier shift
   theorem and orbital-rotation covariance at every sampled frequency.

5. **Small amplitudes and equal-mass backward evaluation.** Replace square roots
   involving vanishing mass asymmetry and PN complex magnitudes with an exact
   positive-input square root, evaluated on a safe inactive branch at zero.
   This removes `0 * inf` backward operations and avoids the inherited magnitude
   helper's positive-value cutoff. At the nondifferentiable boundary the selected
   subgradient is explicitly zero; this does not assert existence of a classical
   derivative with respect to eta at eta=1/4. Positive inputs retain their
   unmodified forward values and derivatives.

6. **Equal-mass body labels.** Extend XHM's mass-based canonical ordering with a
   spin-based tie break at exact equal masses. Reuse the existing `(-1)^m`
   transformation on return. The previous ordering did not canonically handle
   equal masses with unequal spins, so exchanging their labels could fail
   Eq. (2.13). A batched test covers both unequal and equal masses and checks
   finite full-model gradients for the selected branches.

7. **HM merger-ringdown phase.** Apply London Eq. (9) to the D carrier before
   phase-frequency mapping. Write `r = f_RD_22 / f_RD_lm`,
   `w = r * f_damp_lm`, and `c = alpha_5 * f_RD_22`. The changed term is
   `(alpha_4/eta) * r * atan((Mf-c)/w)`; the old term is
   `(alpha_4/eta) * atan((Mf-c)/f_damp_22)`. If their difference is `g(Mf)`,
   the C1-matched correction above the carrier's merger-ringdown join
   `x_c = f_RD_22/2` is `g(Mf) - g(x_c) - (Mf-x_c)*g'(x_c)` and is zero
   below the join. This is equivalent to solving the modified ansatz's
   `alpha_0` and `alpha_1` from the unchanged intermediate phase and derivative.
   Apply the correction at both mapped frequencies and both mapping boundaries
   before constructing Eq. (7)'s offsets. In reference-frequency phasing,
   subtract its value at the carrier's reference as well. The existing global
   affine phase/time gauges are retained rather than independently certified.
   HM's quadrupole QNM helper now returns the D carrier's own frequency and width,
   avoiding two inconsistent quadrupole QNMs in the replacement. Higher-mode
   QNM fits remain unchanged. The Eq. (9) correction is specific to D carriers;
   the optional XAS/custom-carrier paths do not receive this D-ansatz change.

## Validation

Command, run from `/Users/xangma/repos/torchwave`:

```sh
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q tests/test_clean_higher_modes.py tests/test_phenomhm_reconstruction.py tests/test_phenomxhm_reconstruction.py tests/test_imrphenom_hm.py
```

Result: **37 passed in 5.38 s**. `git diff --check` also passed. The thirteen new
cases check published leading-PN amplitudes, the HM transition/mapped envelope,
XHM time and orbital-phase covariance across all supported modes and frequency
regions, time/phase derivatives, small nonzero magnitudes, equal-mass fit
subgradients, and batched body exchange/full-model backward evaluation. The
new HM tests independently reconstruct the full Eq. (9) ansatz and solve its
affine matching coefficients, check all correction-parameter gradients against
finite differences, verify propagation through every higher mode and both phase
offsets, check a reference within merger-ringdown, test phase/slope continuity at
both map joins and the mapped carrier join, and check batched full-model backward
evaluation. The existing derivative-relation test now includes the independently
written analytic Lorentzian derivative change; its accuracy threshold remains
unchanged. Other existing tests continue to check mode support, symmetry limits,
finite strain, mapping connections, and reconstruction behavior. No test
tolerance was relaxed.

## Explicit unresolved dependencies

- **HM is not fully reconstructed.** The inherited extra odd-l phase factor and
  the reference-frequency term using
  the D peak slope were not independently derived in this bounded patch.
  The inherited 43 spin term and the precise PN truncation for `m=l-1` modes
  still require an independent coefficient/truncation audit. All six advertised
  HM modes, including 43, remain present.
- **XHM phase is incomplete.** The inspiral phase lacks the complete
  frequency-dependent PN phase correction and reconstruction of Eq. (4.11).
  Inspiral-to-intermediate connections currently enforce phase values without
  the complete published derivative/alignment construction. Inherited
  mode-dependent phase offsets, including the 21 sign decision at `Mf=0.008`,
  still have no independent derivation here. Old specification references in
  these untouched sections are not evidence supporting them.
- **XHM amplitude calibration is not certified.** Only one nonspinning ancillary
  expression was checked. Normalizations connecting fitted inspiral,
  intermediate, and ringdown quantities need independent reconstruction;
  the intermediate fit-to-envelope conversion is specifically suspect.
  The published unexpanded 21 PN branch and parameter-dependent collocation
  veto/reconstruction rules are not fully implemented. Pointwise polynomial
  fallback behavior also needs comparison with the published interval logic.
- **XHM 32 mixing remains incomplete.** The current hard switch between unmixed
  and mixed strains does not implement the full boundary reconstruction.
  Conjugation and quadrupole spheroidal-normalization factors in the published
  mixing transformation require correction/verification. Its shape and join
  accuracy are not established by the successful covariance test.
- **Conventions and inherited fits remain dependencies.** Unprojected positive
  mode keys store physical-amplitude positive-frequency modes in the carrier's
  `exp(-i phase)` convention, corresponding to the paper's negative-m
  positive-frequency quantity. XHM's returned negative keys are an API
  conjugation convention and should not be confused with physical same-frequency
  support. Carrier phase anchors, remnant/QNM consistency, projection/precession
  convention composition, and the remaining author calibration assets require
  coordinated verification outside these bounded changes.
