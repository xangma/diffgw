# SEOBNRv4 NQC public precision audit

Date: 2026-09-14

Scope: clean-room review of the public calibration material behind Bohé et al.,
arXiv:1611.03703v3, Appendix A (Eqs. A1--A12) and Eqs. (4.9), (4.15). The
clean implementation inspected was `src/torchwave/approximants/eob/nqc.py`.

## Source search and provenance

The primary source located is the author paper release at
[arXiv:1611.03703v3](https://arxiv.org/abs/1611.03703), including its source
archive (`https://export.arxiv.org/e-print/1611.03703v3`). The downloaded
archive contains `paper.tex`, figures, and paper tables; it contains no
ancillary numerical calibration file or machine-readable coefficient release.
The source archive observed on this audit has SHA-256
`6d9f2e99524c9878ea06d31425939bee4e8dab52948878037d75132bcfeac488`.

The paper itself says the model is implemented in LAL and publicly available,
but that statement does not identify a separately licensed, full-precision
calibration artifact. No author ancillary data, DCC technical note, or
supplemental release containing more digits for these Appendix-A and
time-offset fits was found in the public primary-source search. The paper
release therefore supplies equations and printed decimal coefficients only.
No implementation data was imported, and no LAL or other implementation source
was consulted.

## Printed-decimal rounding bounds

For these two families, every printed literal is treated as independently rounded to
the shown last digit (half-unit bound). The reproducible diagnostic
[`seob_nqc_precision_bounds.py`](seob_nqc_precision_bounds.py) evaluates all
interval corners at `nu=.24, chi=0` for the frequency and slope targets. The
expressions are monotone in each interval variable at this point, so corner
enumeration gives the extrema. The JSON output is checked in as
[`seob-nqc-precision-bounds.json`](seob-nqc-precision-bounds.json). These are
local target-fit bounds from printed rounding alone, not bounds on the
downstream waveform.

For frequency, `a3` is positive and `log(a3)` is monotone; the coefficient
weights at this point have fixed signs, so each endpoint is attained at an
interval corner. For slope, the interpolation weights at `nu=.24` are
`(0.0784012544, 0.9215987456, 0.0095219124)` for `(tpl, equal-mass, linear)`; the log
argument is positive and its `.004087` coefficient is included as a rounded
fit coefficient. These fixed-sign weights justify the same corner calculation.

| target family | last printed digit | local interval result |
| --- | ---: | ---: |
| peak frequency (A8--A9), at `.24,0` | 1e-6 | half-width `1.69772e-6` |
| frequency slope (A10--A12), at `.24,0` | 1e-6 | half-width `6.70443e-7` |

The requested screening residuals, `7.0e-7` for frequency and `4.6e-7` for
slope, are below both signed distances from the nominal value to the interval
endpoints. Printed rounding is therefore numerically capable of explaining
residuals of either sign at this point; this is screening only and is not a fit
or attribution. The matching
solve can amplify target perturbations according to its conditioning and
trajectory stencil, so these figures do not bound solved coefficients or a
waveform. The result does not establish that the reference used hidden
precision.

## Decision

### 2026-09-14 implementation update

The published six-decimal helper is retained unchanged. The explicit numerical
compatibility route now evaluates the same public equations with each printed
time-offset coefficient rounded independently to six significant digits before
polynomial evaluation. This is a newly audited contract; the previous
implementation had no such precision change. The source is
[Bohé et al., arXiv:1611.03703v3](https://arxiv.org/abs/1611.03703), and the
functional boundary is recorded in
[reference-audit-contracts.md](reference-audit-contracts.md).

No independently admissible full-precision calibration artifact was found.
The public paper equations remain the clean-room source contract; a future
provider requires an independently identified artifact, version/hash, and
explicit redistribution permission.
