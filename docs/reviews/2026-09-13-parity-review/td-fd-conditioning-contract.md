# Time-domain to frequency-domain compatibility contract

This is a source-free functional handoff from the user-authorized reference
auditor to isolated Luna implementers. It applies to the existing strict
SEOBNRv4 and TEOBResumS cases, whose reference is `SimInspiralFD`. It does
not replace the physical waveform equations. Implement independently using
standard PN timing, Butterworth filter, and discrete Fourier mathematics.
No reference implementation, filter code, or source-only calibration table
is supplied. Reference version and separation are recorded in
[the audit contracts](reference-audit-contracts.md).

## Generate sufficient time support

Use the detector-frame total mass in seconds, `M`, and symmetric mass ratio
`eta`. For the requested lower GW frequency `f_low`, define
`f_safe=min(f_low, 1/(27*pi*M))`. Keep the originally requested frequency for
the high-pass filter. The duration bound is the TaylorT2 timing through 2PN
with the negative terms omitted: retain the Newtonian, 1PN and 2PN timing
terms, and the positive spin overestimate `(226/15)*max(|chi1z|,|chi2z|)*v^3`
in the normalized bracket. Derive the nonspin timing coefficients from the
primary PN equations. This is an allocation bound, not the physical EOM.

Add one orbital period at radius `9M` for plunge and eleven damping times
of the fundamental 22 Berti/Cardoso/Will fit for ringdown, using the total
initial mass. Obtain that QNM fit from their primary paper, Appendix E,
Table VIII (https://arxiv.org/abs/gr-qc/0512160). The spin bound is the
larger of the two initial spin magnitudes and the equal-mass aligned linear
estimate of Tichy/Marronetti (https://arxiv.org/abs/0807.2985), capped at
0.998. Verify the paper's printed coefficients; do not use an alternative
remnant prescription for this support calculation.

The generation start frequency is the inverse Newtonian chirp-time relation
for the duration `1.1*t_bound + t_plunge_ringdown + 3/f_safe`. Generate the
physical TD waveform from this lower frequency, preserving the requested
reference frequency and orbital-phase semantics.

Public-binary diagnostic values at zero spins:

| Masses (solar) | Requested Hz | Chirp bound s | Plunge + ringdown s | Start Hz | First taper s |
| --- | --- | --- | --- | --- | --- |
| 30 + 20 | 20 | 1.7156926398807189 | 0.07484355785826285 | 16.900094009839023 | 0.3215692639880719 |
| 1.4 + 1.4 | 30 | 56.080416766565115 | 0.00419123924006272 | 28.430883710960526 | 5.708041676656512 |

These values are validation samples, not runtime constants.

### Binary64 unit conventions

The September 14 public-stage diagnostic shows that one binary64 step in
the generation frequency can change the public TEOB waveform at relative
L2 of order `1e-4`. Algebraically equivalent timing expressions therefore
need a declared numerical convention. A follow-up source audit by the
coordinator identified the following unit boundaries for an isolated
implementation from the published TaylorT3 Newtonian relation:

- Form the component masses in SI, total mass, reduced mass, and symmetric
  mass ratio in that order.
- Express frequency as the Newtonian TaylorT3 frequency prefactor times
  the dimensionless inverse-time factor to the `3/8` power.
- The frequency prefactor converts SI mass using `G/c^3`; the dimensionless
  time factor uses the independently evaluated ratio of solar mass in
  seconds to solar mass in kilograms. These conversions are physically
  equivalent but need not have identical binary64 intermediate values.
  In the inverse-time numerator, convert five times the total SI mass to
  seconds. Evaluate the frequency prefactor separately, then multiply it
  by the inverse-time factor raised to `3/8`.

Derive the coefficients and time variable from the primary PN equations;
do not select operation permutations or numerical nudges by waveform error.
This contract does not assert that Torch elementary operations will be
bitwise identical to the public binary across platforms. Validate and
report discrepancies separately from waveform acceptance.

## Condition the real polarizations

First record the number of trailing samples that are exactly zero in both
polarizations. Taper the beginning over `0.1*t_bound + 3/f_safe`, using a
half-cosine rise: at sample offset `j`, the weight is
`(1-cos(pi*j/n_taper))/2`, where duration-to-sample conversions round to
nearest with positive half-integers rounded upward. The first untouched
sample has offset `n_taper`.

Apply an eighth-order Butterworth high-pass at the originally requested
`f_low`, with specified final power transmission 0.99 there. Use a bilinear
transform with prewarping. The dimensionless analog corner is
`tan(pi*f_low*dt)*(1/sqrt(0.99)-1)^(1/16)`.
Use four real conjugate-pole sections, in increasing positive damping angle
`theta=(j+1/2)*pi/8`, `j=0..3`. Each section is applied in both time directions
before proceeding to the next section. Each directional pass starts with
zero filter state; there is no reflection padding or steady-state startup.
Independently derive the second-order coefficients from the Butterworth
poles and bilinear transformation.

Remove the previously counted trailing zero padding only after filtering.
Then taper one cycle at `f_safe` from the beginning and one cycle at the
Schwarzschild ISCO GW frequency `1/(6^(3/2)*pi*M)` from the end. Each taper
uses at least four samples; skip this second stage for fewer than eight
samples. The beginning uses the same half-cosine convention. The end would
reach zero at the next sample beyond the data: the last retained sample has
weight `(1-cos(pi/n_end))/2`, and a sample `j` steps backward from the next
sample has weight `(1-cos(pi*j/n_end))/2`. Reject invalid taper durations
rather than allowing out-of-range indexing in native code.

The coordinator independently derived biquad coefficients from these filter
parameters and compared random vectors through the public filter API. With
zero states, lengths 4096, 4096, 13, and `(dt,f_low)` of `(1/2048,20)`,
`(1/1024,30)`, `(.002,40)`, relative L2 errors were respectively
`6.78e-14`, `1.39e-14`, and `1.19e-15`. No parameter was fitted.

## EOB mode-to-polarization frame

The v4 mode API and direct `SimIMRSpinAlignedEOBWaveform` API use an observer azimuth
`varphi = pi/2 - phic` for the spin-weighted harmonics. For a dimensionless
nonprecessing 22 pair, form

```
H = sqrt(5/(64*pi)) * M_seconds*c/(distance_Mpc*Mpc)
    * [(1+cos(iota))^2 * h22 * exp(2i*varphi)
       + (1-cos(iota))^2 * h2m2 * exp(-2i*varphi)]
h_plus = Re(H), h_cross = -Im(H), h2m2 = conj(h22)
```

This is a frame definition, not a fitted overall phase. In particular both
modes acquire a minus sign relative to the zero-azimuth chart at `phic=0`.
The coordinator compared the public binary's mode and polarization APIs for
80+60 solar masses, at four inclination/phase choices; the formula agrees
within 3.1e-16 relative L2 for both polarizations. See
[projection.json](verification/eob-conditioned-root-review/projection.json).
The generic `SimInspiralChooseTDWaveform` and `SimInspiralFD` interfaces then
rotate the polarization basis by `-pi/2`, reversing both polarization signs.
The adapter targets this **generic** frame. Consequently its net 22-mode
weights are `exp(-2i*phic)` and `exp(+2i*phic)`, exactly the original adapter's
net formula. Do not add another minus sign based on a direct-EOB API comparison.
The two API frames must be distinguished in diagnostics. Do not reanchor
orbital phase at the frequency peak. Physical epoch and explicit `tc` remain
separate as described below. Test both transformations, including nonzero
`phic`, and explain the net generic frame in the API.

## Fourier sampling and epoch

For an explicitly requested `delta_f`, round `f_max/delta_f` to the nearest
integer; if it is not a power of two, increase it to the next power of two.
Use that Nyquist frequency and `dt=1/(2*f_Nyquist)`. The Fourier length is
`N=2*f_Nyquist/delta_f`. Keep the final `N` conditioned TD samples, padding
with leading zeros when necessary. Thus a coarse frequency spacing can
truncate the early inspiral; do not substitute a full-duration quadrature
and call it the same reference interface. For automatic frequency spacing,
the reference chooses a power-of-two length strictly greater than the
nonzero TD length, including when that length is already a power of two.

Compute `dt*rfft(h)` for each real polarization. The physical epoch is
metadata and is not an additional phase multiplier in the returned array.
A public FFT check with an identical ramp at epochs 0 and -1.23456789 seconds
confirmed identical arrays, agreeing with the scaled ordinary rFFT within
`1.74e-18`. Explicit user coalescence-time shifts remain a separate waveform
parameter. Keep that distinction from the physical-epoch direct integral
used in the existing raw-TD diagnostics.

Validate synthetic filters/tapers/FFT independently first; then use public
TD outputs strictly as test fixtures to isolate conditioning from dynamics.
Production must generate its own TD waveform and must not import LAL.
Keep the existing strict catalog frequencies and thresholds unchanged.
