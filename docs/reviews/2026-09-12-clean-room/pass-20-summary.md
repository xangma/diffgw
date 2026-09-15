# Pass 20: SEOBNRv4 source-boundary audit

Date: 2026-09-13. Three primary-source audits and independent Luna reviews
examined the remaining bounded SEOBNRv4 phase, remnant, NQC, and attachment
questions. No source-complete runtime correction was found, so this pass changes
documentation only. Full LAL parity remains unfinished.

## Decisions

- The native pre-NQC factorized `(2,2)` phase retains the published Newtonian
  sign gauge, tail-radius conversion, and Taracchini A16 residual. Bohé's
  displayed spin term is inconsistent with the paper's stated PN order and
  test-particle limit if its energy symbol is read literally. The total-energy
  interpretation remains declared rather than silently promoted to a unique
  SEOBNRv4 default. Older Pan and Taracchini spin terms also remain explicit
  choices. See the [primary audit](pass-20-eob-phase22-primary.md) and
  [independent review](pass-20-eob-phase22-independent-review.md).
- The final-mass interpolation and all three Hofmann final-spin coefficient
  blocks match the primary sources. SEOBNRv4 cites the Hofmann fit without
  identifying `(1,2)`, `(3,3)`, or `(3,4)`, so `fit_order` remains required.
  See the [primary audit](pass-20-eob-remnant-primary.md) and
  [independent review](pass-20-eob-remnant-independent-review.md).
- The isolated NQC equations and attachment path match the published five
  conditions and single-fundamental C1 merger ansatz when the caller supplies
  the continuous mode, positive GW frequency, and explicit remnant/QNM
  metadata. The default `SEOBNR_ODE` path instead joins at the last trajectory
  sample and appends a PhenomD exponential. It is therefore still a prototype,
  not a source-complete SEOBNRv4 waveform. See the
  [primary audit](pass-20-eob-attachment-primary.md) and
  [independent review](pass-20-eob-attachment-independent-review.md).

## Verification and next dependency

The combined source-derived EOB suites passed **149 tests with 4 skipped in
1.95 s**. They cover the `(2,2)` phase and amplitude building blocks, remnant
fits, NQC solve, attachment, merger/ringdown join, and prototype waveform.
There were no runtime or test changes, so the Pass 19 ordinary and frozen
verification remains current: **1,142 passed with 9 skipped**, **11/21 strict
catalog**, **4/13 holdouts**, **3/34 explicit components**, and **4/4 mode-sum
controls**. No acceptance threshold, calibration coefficient, reference case,
or parity harness changed.

The next EOB implementation dependency is a declared target specification for
the residual spin-phase inheritance and Hofmann block, followed by wiring the
continuous factorized mode through the orbital-frequency peak, NQC, explicit
remnant/QNM selection, and the C1 merger ansatz. That generator must then pass
independent SEOBNRv4 validation before the existing ROM can be retrained. A
complete calibrated TEOBResumS implementation remains a separate track.

The user-deferred 48-worker numerical-relativity run remains stopped and is not
required for this source audit.
