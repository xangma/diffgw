# Bounded parity strategy experiment

This protocol was written before fitting the new complex models or inspecting
their validation residuals. Existing parity gates, fits, manifests and runtime
implementations remain independent of this diagnostic experiment.

## Independent waveform experiment

- Training: the five `training` families in the v3 qualified manifest, with
  their same-record two-resolution N2/N3 controls. Existing seven phase
  benchmark cases are previously examined regression cases, never selection
  inputs. Fresh simulation families, when qualified under the unchanged
  amplitude quality policy, are the final holdouts.
- Domain: nonspinning, nominal q=1.5 through 8, and the same physical frequency
  interval `0.95*f_ring,33 <= Mf <= 0.3`. Both 22 and 33 are evaluated on that
  interval; this is not their full waveform or detector-band validation.
- Inputs: q and independently computed 33 QNM frequencies from each NR
  remnant. This is a conditional spectral surrogate; it does not supply a
  q-to-remnant model. No LAL values, tables, source, or residuals enter fitting.
- Gauge: set time zero at the sampled |h22| peak. One common orbital rotation
  sets h22 there real positive. Resolve its remaining pi orbital ambiguity
  by requiring the imaginary part of rotated h33 there to be nonnegative.
  The rotation is applied as exp(-i*m*phi) to both modes, before the frozen
  conjugated FFT. No continuous per-mode alignment is fitted for scoring.
- Representation: 512 uniformly spaced bins in the affine coordinate mapping
  the declared physical band to [0,1]. Divide by the analytic leading 22
  Fourier amplitude, and by delta=(q-1)/(q+1) for 33. Divide both modes by
  the dimensionless damped-oscillator factor
  `f_damp/(f_damp + i*(f-f_ring))`. The 33 QNM provides a common coordinate
  and conditioning factor, not a claim that it is the 22 QNM. Equal-mass
  evaluation is outside the experiment. Fit training-derived per-mode RMS
  scales anew in every cross-validation fold.
- Candidate: joint complex uncentred SVD, rank in {1,2,3}, polynomial degree
  in {1,2} in x=4*eta-1. Choose by mean per-family, equally weighted per-mode
  raw complex relative L2 in leave-one-training-family-out validation.
  Report endpoint extrapolation folds separately. Ties sort by rank, degree.
- Comparator: existing Eq.42 amplitude ansatz, sigma33=1.3, quadratic
  coefficient surfaces refitted on each training fold. Supply its missing
  phase using a training-only polynomial interpolation of unwrapped phase
  in the same normalized representation. Select phase degree {1,2} on
  training folds only. The comparator's 22 mode uses log-amplitude/phase
  interpolation. This is a new diagnostic completion of an amplitude-only
  baseline, not the unimplemented London mapped-carrier phase model.
- Quality: before selection, record high/low N3, high N2 and the prior dt=.05M
  and end-taper=40M sensitivities, plus phase RMS diagnostics. Retain all five
  previously amplitude-qualified training rows for this exploratory comparison.
  Flag phase sensitivity >.01 rad (resolution/extraction), >.002 rad
  (preprocessing); these prevent a claim of qualified scientific calibration,
  rather than silently selecting a favourable subset. Sensitivities are not
  rigorous error bars. Do not tune weights to these five cases.
- Freeze selected models and source/input hashes before any final evaluation.
  Report 22/33 raw complex L2, amplitude L2, common-gauge relative-phase error,
  and an equal-weight coherent sum as a diagnostic, without claiming a
  detector strain or astrophysical mismatch. Report a full-rank reconstruction
  error to distinguish representation compression from parameter interpolation.
- Check query gradients against centred finite differences. Query differentiation
  covers the conditional normalized spectrum; it does not validate physical
  end-to-end gradients through the NR remnant or data preprocessing.
- No runtime promotion. A win requires lower worst-case and median heldout
  complex errors than the completed baseline, no new >1% 33 amplitude errors
  on interpolation cases, and explicit comparison with data sensitivities.
  Absence of fresh qualified holdouts leaves the prospective validation open.

## EOB experiment

Replay cached TEOB trajectories with independently controlled event time and
auxiliary grid. Record NQC coefficient changes, matrix conditions, matching
indices and waveform errors. Perturb controls separately; do not attribute a
cause or implement a runtime fix unless the controlled replay supports it.
