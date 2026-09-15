# NR $(3,3)$ phase benchmark: method review

This note fixes the conventions and diagnostics for the nonspinning NR $(3,3)$
phase benchmark.  It is a methodology review only; no NR waveform, LAL output,
or fit was evaluated while writing it.

## Public source facts

* SXS publishes spin-weighted spherical-harmonic mode weights, with files that
  contain `(t/M, Re[R_areal Q_lm], Im[R_areal Q_lm])`; its documentation says
  that the strain files are the $h_{\ell m}$ (or, where selected, $\Psi_4$
  mode) coefficients.  The public documentation also distinguishes raw finite-
  radius data from inertial and center-of-mass corrected data:
  [SXS waveform documentation](https://data.black-holes.org/waveforms/documentation.html),
  [SXS waveform API](https://sxs.readthedocs.io/en/stable/api/waveforms/).
* SXS describes extrapolated orders, outermost extraction, and CoM corrections
  as separate public waveform products.  It explicitly recommends comparing
  extrapolation orders to assess extrapolation error.  Its catalog paper gives
  the strain-mode construction and numerical-error discussion:
  [Boyle et al., SXS catalog (2019)](https://arxiv.org/abs/1904.04831),
  [published PDF](https://pure.mpg.de/pubman/item/item_3047792_4/component/file_3047793/1904.04831.pdf).
* The SXS tutorial warns that a ringdown slice can acquire “numerical garbage”
  if extended too far, and notes that the catalog strain includes memory.  This
  is evidence for a tail-window check, not a claim that a particular case is
  corrupted: [SXS waveform tutorial](https://sxs.readthedocs.io/en/stable/tutorials/04-Waveforms/).
* The published IMRPhenomXHM conventions explicitly state equatorial symmetry,
  the $e^{-i2\pi f t}$ Fourier transform, the concentration of $m>0$ modes at
  negative frequency, and the positive-frequency $h_{\ell,-m}$ representation
  (Sec. II.1, Eqs. (4)--(7)): [García-Quirós et al., IMRPhenomXHM](https://arxiv.org/html/2001.10914v2#S2.SS1).
  Public LALSimulation APIs expose positive-frequency mode/polarization series
  and identify the inputs as spin -2 spherical-harmonic modes; the release
  adapter still needs an interface/epoch check:
  [LALSimInspiral API](https://docs.ligo.org/lscsoft/lalsuite/lalsimulation/_l_a_l_sim_inspiral_8c.html),
  [LALSimIMRPhenomXPHM API](https://docs.ligo.org/lscsoft/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x_p_h_m_8c.html).
* Noise-weighted waveform comparisons and maximization over the extrinsic time
  and phase are standard in NR/analytic comparisons; see Secs. IV.1 and V.3 of
  [Santamaria et al. (2010)](https://arxiv.org/abs/1005.3306).  Its discussion
  also makes clear that a PSD introduces total-mass dependence.

## Fourier and $(m,-m)$ convention

Use one explicitly stated transform throughout:

\[
  \tilde x(f)=\int x(t)e^{-2\pi i f t}\,dt.
\]

For a nonprecessing, equatorially symmetric source, the published IMRPhenomXHM
conventions (Sec. II.1, Eqs. (4)--(7)) give the mode and Fourier relations

\[
  h_{\ell,-m}(t)=(-1)^\ell h_{\ell m}(t)^*.
\]

Thus $h_{3,-3}(t)=-h_{33}(t)^*$ under the convention used here.  With the transform above,
$\mathcal F[x^*](f)=\tilde x^*(-f)$, so forming the positive-frequency
representation of the negative-$m$ mode from stored $h_{33}$ requires the
conjugated time series and the $(-1)^3=-1$ parity factor.  This is the
mathematical reason for the planned `FFT(conj(h33))`, followed by `-1`; it is
not a choice to be made by whichever sign gives the smaller fit residual.

The epoch correction must be derived with the same convention.  If
$x'(t)=x(t-\Delta t)$, then $\tilde x'(f)=e^{-2\pi i f\Delta t}\tilde x(f)$.
Therefore the planned factor
`exp[-2*pi*i*f*(fft_epoch-h22_peak_time)]` is consistent when the raw FFT is
referenced to `fft_epoch` and the desired origin is the $h_{22}$ peak.  The
implementation records this exact operation in
[`scripts/nr33_phase_data.py`](../../../scripts/nr33_phase_data.py).

Before scoring, run a convention-only adapter check: (i) FFT a manufactured
positive/negative-frequency damped sinusoid, (ii) verify that SXS columns are
assembled as `Re + i Im`, (iii) verify the SXS nonspinning $\pm m$ identity on
one public record, and (iv) check the release adapter's positive-frequency
argument and epoch packaging against the documented LAL interface.  The check
must be recorded as a convention test; it must not select a sign by minimizing
a physical-model residual.  The published equations fix the sign mathematically;
the adapter check only verifies that the release interface preserves it.

## Alignment and reported metrics

Center each waveform independently at its own sample of maximum $|h_{22}|$: for
each resolution and extrapolation order, that waveform's own $h_{22}$ peak is
$t=0$.  This makes raw complex differences a post-peak-centering diagnostic; it
does not assert a shared global SXS gauge or prohibit fitting the remaining
constant phase and time offsets.

For a diagnostic phase comparison, retain the predeclared band and mask samples
where the NR amplitude is below $10^{-3}$ of its band peak.  On the retained
samples use $w(f)=|h_{33}^{\rm NR}(f)|^2$ and fit only a constant phase and an
affine time phase.  Report the fitted values and weighted phase RMS; the
pre-registered $0.002$ rad preprocessing tolerance and $0.01$ rad diagnostic
resolution are acceptance/resolution settings, not rigorous numerical-error
bounds.

Separately report a complex normalized overlap over the full ringdown band,
with constant phase and time shift maximized only in the fixed $\pm200M$ time
window.  For detector weighting use

\[
 (a|b)=4\,\Re\int_{f_{\min}}^{f_{\max}}
       \frac{\tilde a(f)\tilde b(f)^*}{S_n(f)}\,df,\qquad
 {\cal M}=1-\frac{(a|b)}{\sqrt{(a|a)(b|b)}}.
\]

Use the public LAL `SimNoisePSDaLIGOZeroDetHighPower` model at total masses
20, 60, and 120 $M_\odot$, retaining physical frequencies $f\ge20$ Hz after
converting the dimensionless $Mf$ band.  Report the three masses separately:
the PSD weighting and the conversion of a fixed $Mf$ interval to Hz are
mass-dependent.  A complex single-mode overlap is not automatically a detector
polarization mismatch; if $h_+$/$h_\times$ are constructed, state the
inclination and polarization convention as an additional choice.

The frozen benchmark band is $0.95 f_{\rm ring,33}\le Mf\le0.3$.  Use the
common NR grid and preprocessing (`dt=0.1M`, start `500M`, end `20M`, padding
factor 4) for all cases.  A finite-duration/tapered FFT can leak power into a
high-frequency tail; record the actual retained bins after applying both the
band and physical-frequency cuts.

## NR reliability and required sensitivity checks

The SXS public guidance gives a rough mode-amplitude trust floor of order
$10^{-5}$ of the dominant mode and says that comparing extrapolation orders is a
way to assess extrapolation behavior.  The benchmark's $10^{-3}$ phase mask is a
separately defined diagnostic support rule; its normalization and the SXS
dominant-mode guidance are different, so their numerical thresholds are not
directly comparable and the mask does not prove accuracy.
The catalog/tutorial evidence supports the following separate checks:

1. Compare N2 versus N3 and the two highest available resolutions.  Treat their
   spread as an observed sensitivity, not a one-sided error bar.
2. Repeat the phase and overlap summaries with nearby tail end times and report
   the change.  Late ringdown can be dominated by numerical noise; higher
   extrapolation order can improve inspiral while worsening late ringdown, and
   the outermost extraction can sometimes be preferable there.  These are
   documented possibilities, not a demonstrated cause of the present result.
3. Keep finite-radius/extrapolation sensitivity, finite-resolution convergence,
   CoM/frame choices, taper/padding, and memory-baseline choices as separate
   columns in the audit.  A resolution change cannot diagnose an extrapolation
   problem by itself.
4. Report amplitude support and phase RMS alongside mismatch.  A small overlap
   mismatch can coexist with large phase excursions where the mode has negligible
   amplitude; conversely, a short high-frequency tail can receive substantial
   detector weight for some total masses.

## Interpretation limits

The seven holdouts (`q=3, 3.5, 4.5, 5.5, 6, 7, 10`) test interpolation/extrapolation
of the frozen model, while the NR-versus-NR checks test data preparation and
numerical sensitivity.  Neither establishes a global NR error bound.  The
specified ringdown band may contain little reliable information after the tail
mask and finite-duration FFT, so any disagreement there should first be
localized in amplitude support, extrapolation order, resolution, and tail
window before attributing it to the analytic phase model.  The convention tests
above are prerequisites; no sign or phase gauge may be selected from the final
fit residual.
