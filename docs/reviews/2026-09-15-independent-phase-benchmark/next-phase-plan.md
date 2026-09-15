# Next phase: NR $(3,3)$ phase calibration plan

This plan follows the completed seven-case benchmark while keeping all seven
benchmark cases as holdouts: $q=3,3.5,4.5,5.5,6,7,10$.  The current training
manifest supplies five metadata-screened rows, one selected N3 waveform per
mass ratio: SXS:BBH:2331 ($q\simeq1.5$), 1166 ($q\simeq2$), 2485 ($q\simeq4$),
3619 ($q\simeq5$), and 2707 ($q\simeq8$) ([manifest](../2026-09-15-independent-calibration-v3-search/nr33-qualified-manifest.json)).
Those rows are eligible training inputs by the existing data screen; phase
qualification is still required before fitting.

## Data and qualification

1. For each of the five training families, preserve same-record metadata and
   obtain N2/N3 plus the two highest available resolutions.  Check the
   extrapolation order, inertial/CoM frame, time units, sidecar group, and
   waveform hashes before any phase fit.  A single selected N3 row cannot
   establish phase convergence.
2. Use [`nr33_phase_data.py`](../../../scripts/nr33_phase_data.py) to retain the
   complex mode, apply the fixed positive-frequency convention, and center each
   waveform at its own $|h_{22}|$ peak.  Record that raw complex differences
   are post-peak-centering diagnostics; retain fitted constant phase and time
   offsets as measured gauge/shape alignment parameters.
3. Phase-qualify only the declared band and preprocessing: the frozen
   $0.95f_{\rm ring,33}\le Mf\le0.3$ interval, common $dt=0.1M$, start
   $500M$, end $20M$, and padding factor four.  Repeat with nearby tail cuts.
   Use the amplitude support mask and report retained support, N2/N3 spread, and
   resolution spread.  The existing `.002` rad preprocessing and `.01` rad
   diagnostic settings remain diagnostics, not numerical error bounds.

The seven holdouts passed the current phase checks. Preserve those checks; the
holdouts must not influence ansatz choice, coefficient fitting, order selection, or
alignment-window changes.  In particular, q3 and q6 remain tests of a phase
prediction; their NR resolution/extrapolation and finite-tail sensitivities limit
confidence in attributing the remaining phase disagreement to the model.  A holdout residual is not a
universal NR phase-error estimate.

## Public-math ansatz and blockers

The clean interfaces currently expose useful algebra but no complete calibrated
33 phase model.  [`imrphenom_hm_phase_clean.py`](../../../src/torchwave/approximants/imrphenom_hm_phase_clean.py)
contains the published London frequency map and Eq. (14) correction for
arbitrary positive $m$, so it can provide an $m=3$ baseline when the caller
supplies the carrier phase, QNM frequencies, and required carrier quantities.
[`xhm_phase_clean.py`](../../../src/torchwave/approximants/xhm_phase_clean.py)
implements the published four-condition spheroidal **32** ringdown primitive;
it does not provide a 33 primitive or a parameter-space coefficient map.  The
production XHM module contains a structural 33 phase branch, but it calls
`XHMParameterFits`; its release fit tables and coefficient bodies are outside
this clean-room calibration.

Therefore the next implementation must first freeze a 33 ansatz and its input
contract.  A defensible public-math route is: use the London $m=3$ mapped
carrier as the baseline, define a low-dimensional 33 residual phase (or a
clean transcription of the published no-mixing 33 branch), and fit only its
coefficients to the five training families.  The coefficient surface should
use a declared intrinsic coordinate such as $\eta$ (with spin inputs fixed by
the nonspinning selection), and its basis/order must be chosen using training
rows only.  The phase model must return the same positive-frequency complex
mode convention as the extraction module and expose phase plus derivative
without importing LAL output or release coefficients.

The immediate interface blockers are therefore (a) no clean 33 phase evaluator,
(b) no caller-owned schema for 33 derivative targets/residual coefficients,
(c) no phase-specific qualification artifact for the five training rows, and
(d) no frozen rule yet for whether a tail-sensitive row is excluded or merely
reported.  Resolve these before generating a calibration table.

## Scoring and acceptance sequence

After training-only ansatz selection, score every holdout with
[`nr33_phase_metrics.py`](../../../scripts/nr33_phase_metrics.py): raw,
constant-phase, and affine time/phase diagnostics, followed by the complex
normalized detector-weighted overlap over the full band.  Use the fixed
$\pm200M$ alignment bound and the preregistered PSD/mass settings.  Report
per-case phase support, fitted offsets, N2/N3 and resolution sensitivities, and
tail-cut sensitivity beside mismatch.  Only after this frozen holdout report
may any optional ansatz revision be proposed as a new calibration round.
