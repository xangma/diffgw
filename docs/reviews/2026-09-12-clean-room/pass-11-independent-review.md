# Pass 11 independent review and integration decisions

GPT-5.6 Luna implementers and separate Luna reviewers worked from primary papers
and native TorchWave code. The primary agent verified source passages and
challenged two normalization claims before integrating the final changes.
Public waveform comparisons were kept separate from implementers and reviewers.

## NSBH

Zappa et al., arXiv:1903.11622v2, Eq. (1), Eqs. (4)–(6) and Table I define the
tidal polynomials using the initial BH spin. The parent checked the downloaded
paper, including both coefficient rows. Using `chi1` corrects the prior
mass-weighted spin variable. The unchanged two-spin BBH baseline is an API
compatibility extension outside the paper's nonspinning-NS calibration; it is
not newly calibrated. Tests cover the calibrated zero-NS-spin case and retained
nonzero-NS-spin behavior.

## XHM normalization

The initial candidate added `N = sqrt(16*pi/5)` to the carrier ratio supplied
to spherical/spheroidal mixing. This was rejected before post-change oracle
verification. The first independent review identified missing production-flow
tests but initially accepted the multiplier. The parent challenged cancellation
of the numerator and denominator; the reviewer then confirmed double counting.

The physical spherical carrier is `N * h22_face_on`, and the paper's leading
mode amplitude is `N * a0_native`. Consequently Eq. (4.6) gives

```text
H22 = abs(N * h22_face_on) / (N * a0_native)
    = abs(h22_face_on / a0_native).
```

The existing final reconstruction multiplies the mixed normalized amplitude by
`N * a0_native`. Adding another `N` before mixing would apply that conversion
twice to the carrier contribution. The final implementation must preserve the
original ratio, with a production regression test. This is a verified convention
and test improvement, not a new waveform normalization correction.

## MSA remnant

The first candidate inferred `Lfin = Mfin² * a_parallel - S_parallel` and divided
`sqrt(<Sfin²>)` by `Mfin²`, selecting that result automatically for MSA versions.
Earlier equations in Pratten arXiv:2004.06503v2 support this vector-identity
variant, but its explicit FinalSpinMod3 Eq. (4.29) instead divides by initial
`M²`. Setting total `M=1` does not resolve the discrepancy: `Mfin=1-Erad` is
smaller. In the aligned limit the two results differ by `(Mfin/M)²`.

The signed-spin paragraph supports the sign of the aligned remnant, but does
not specify a tie-break when that remnant is exactly zero and the precessing
magnitude is nonzero. The first candidate also evaluated the NNLO fallback
eagerly on MSA rows, potentially raising its sign error before strict MSA
versions reported their own invalid initialization.

Automatic remnant selection and the normalized/signed Kerr helper are therefore
deferred. The bounded addition exposes the existing averaged couplings and
computes the source-defined `<Sfin²>` for an explicitly supplied `Lfin`. This
requires neither a mass-denominator choice nor an unpublished sign tie-break.
High-level XP/XPHM remnant behavior remains that of pass 10.

These decisions follow source algebra and API control flow, not waveform
residuals. Focused candidate tests passing did not establish the rejected
normalizations. Final tests and source hashes are recorded separately.
