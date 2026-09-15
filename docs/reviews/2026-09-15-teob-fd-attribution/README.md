# TEOB continuous matching: frequency-domain decision

## Decision

**Do not promote continuous matching to the parity path.** The previous
continuous-matching experiment demonstrated better numerical convergence on
one time-domain comparison. The exact frequency-domain acceptance case and
three additional supported cases show worse LAL agreement. Numerical
convergence and reproduction of LAL's sampled matching convention are separate
objectives.

The better parity workflow is to preserve the audited grid convention, freeze
the accepted trajectory, and compare one stage at a time using the full
acceptance harness. The new scripts make that experiment reusable without
rerunning the trajectory. No threshold, fitted alignment, physical fit, or
public dispatch was changed.

## Unchanged acceptance checks

All cases use zero spins/tides, 100 Mpc, face-on orientation, zero phase/time,
and 0.5 Hz frequency bins. The exact catalog case is 1.4+1.4 solar masses,
30–500 Hz. Full polarizations, PSDs, detector projections, finite values, DC,
and band checks use the existing strict parity harness. Exact bin selection
and the public API's `[flow,fhigh)` mask are applied by a diagnostic wrapper.

| Masses | Requested band, Hz | Grid plus / cross relative L2 | Continuous plus / cross relative L2 |
|---|---:|---:|---:|
| 1.4+1.4 (catalog) | 30–500 | 2.8103e-4 / 1.5607e-3 | 3.4962e-4 / 1.8926e-3 |
| 30+20 (existing reference) | 20–1024 | 3.2716e-4 / 3.2701e-4 | 5.0578e-3 / 5.0505e-3 |
| 20+20 | 25–512 | 1.0304e-4 / 8.2539e-5 | 3.9737e-3 / 4.0017e-3 |
| 1.6+1.2 | 30–500 | 2.8977e-4 / 3.3659e-4 | 3.5520e-4 / 4.2743e-4 |

Every supported case fails the unchanged `1e-6` complex-relative-L2 gate.
The predeclared 40+10 case is outside the implemented published NQC fit domain
(`0.16 < nu <= 0.25`); it failed explicitly and was not extrapolated or silently
replaced. These cases are a bounded validation set, not full domain coverage.

The grid comparisons reuse the exact accepted trajectory and original mode
from each continuous run. Reconstructing the continuous output reproduces the
adapter FD arrays exactly for the catalog and 30+20 controls.

## What caused the regression

The factorial control changes NQC matching and ringdown attachment separately:

| 30+20 policy | Plus relative L2 | Cross relative L2 |
|---|---:|---:|
| Grid NQC, grid ringdown | 3.2716e-4 | 3.2701e-4 |
| Continuous NQC, grid ringdown | 5.1278e-3 | 5.1358e-3 |
| Grid NQC, continuous ringdown | 4.1514e-4 | 4.3873e-4 |
| Continuous NQC, continuous ringdown | 5.0578e-3 | 5.0505e-3 |

The catalog factorization gives the same attribution: changing only NQC
reproduces almost all of the regression. Ringdown attachment is secondary.

The existing grid solve uses `searchsorted(aux_time,t_nqc,right=True)-2`.
Its actual evaluation knot precedes the continuous target time by 0.145 M
(30+20), 0.102 M (20+20), 0.143 M (catalog), or 0.080 M (unequal BNS).
Evaluating continuous spline derivatives at the grid's actual knot leaves
only 0.187% and 0.080% of the phase-coefficient gap for the two BBH cases.
For the BNS cases it leaves 42% and 56%, so derivative and resampling semantics
also matter there. This is a coefficient diagnostic, not a fitted waveform
correction or a claim that all residual error has been explained.

Evidence: each case's `matching-controls.json` and the aggregate
[`nqc-time-controls.json`](nqc-time-controls.json).

## Validation and support controls

### Tail support

The legacy extension has 1500 auxiliary knots. Refining their spacing from
0.5 to 0.05 M changes the tail duration from 750 to 75 M. That also moves the
conditioner's final crop, even when the removed tail has negligible energy.
It therefore cannot establish convergence of a complete FD waveform.

Added optional `support_end` to the native ringdown assembler. It specifies an
absolute physical endpoint, validates the extension, and preserves the existing
1500-knot default, including metadata. No public dispatch or default changed.
The experiment fixes this endpoint at the last accepted trajectory time plus
150 M, independently of public samples, and holds the corrected NQC waveform
and peak fixed.

For both 1.4+1.4 and 30+20, spacings 0.05, 0.1, 0.2, and 0.5 M then produce
bitwise identical complete TD modes and conditioned FD arrays. With legacy
support, non-reference spacings instead differ from the fixed 0.1 M output by
relative FD L2 of 1.16–1.72. The fixed-support result equals the existing 0.1 M
output in these two cases; it improves the validity of convergence experiments,
not their LAL parity score. This control isolates ringdown support only, not
convergence of the trajectory or NQC solve.

Evidence: [catalog tail control](catalog/tail-controls.json),
[30+20 tail control](bbh_reference/tail-controls.json).

### Regression

All 68 focused tests pass. The immutable full parity snapshot remains 17/21
catalog and 11/13 holdout: **28/34, with the same six failures**. Both result
JSON files are byte-identical to the previous frozen regression. Source and
snapshot hashes stayed unchanged during the run. This is regression preservation,
not completion of parity. Details: [validation.json](validation.json).

### Conditioning and attachment

The native conditioner applied to matching public TD data reproduces public
`SimInspiralFD` to plus/cross relative L2 of 2.75e-15 / 2.00e-15 for the exact
catalog case, and 2.66e-13 / 2.55e-13 for 30+20. The catalog TD fixture was
generated freshly from the installed public binary at the exact support start;
the 30+20 fixture is the previously recorded matching request. No public source
was inspected. Native TD-to-FD cache replay is bitwise exact on all four cases.

The Fourier epoch factor is only a diagnostic counterfactual and worsens the
comparison; it is not applied. These results rule out the conditioner as the
source of the observed gap in the paired cases. They do not establish native
waveform parity.

The split is located on the uniform output grid using the physical attachment
time, rather than the joined auxiliary grid's index:

| Case | Pre/post samples | Plus pre/post relative L2 | Cross pre/post relative L2 |
|---|---:|---:|---:|
| Catalog | 64289 / 2 | 1.9252e-4 / 1.4332e-2 | 1.0351e-3 / 6.7807e-2 |
| 30+20 | 4009 / 78 | 2.9509e-3 / 2.3503e-2 | 2.4204e-3 / 2.9081e-2 |

These compare the continuous candidate with public TD at identical elapsed
indices, without fitted alignment. Catalog pre-attachment samples account for
99.46% / 99.91% of the total squared residual norm; only two post-attachment
samples exist at its requested sampling rate, so they give little information
about ringdown. The BBH split retains 42.16% / 28.59% before attachment. Segment
relative errors use each segment's own reference norm and cannot be added.
The controls establish residuals on both sides; they do not isolate a single
physical coefficient responsible for the grid baseline's error.

Evidence: [catalog controls](catalog/controls.json),
[30+20 controls](bbh_reference/controls.json), and
[catalog public TD provenance](catalog/public-td.json).



## Next parity target

Keep sampled NQC matching for strict LAL reproduction. Reuse the saved grid
case and isolate its remaining derivative, resampling, and pre-attachment
residuals on identical states and support. The time-isolation experiment
explains the continuous-policy regression; it does not yet identify a justified
correction for the grid baseline's remaining error. Further tightening of an
ODE tolerance or adding an epoch phase factor is not supported by these controls.

## Reproduce

Use `/Users/xangma/miniconda3/envs/pycbcorig313/bin/python` with `PYTHONPATH=src`.

1. `scripts/teob_continuous_fd_attribution.py --case catalog` generates and caches all native stages, then runs the unchanged gate.
2. `scripts/teob_fd_matching_controls.py --case catalog` factors the two matching policies on frozen inputs.
3. `scripts/teob_nqc_time_controls.py` isolates matching-time versus derivative changes on the four supported cases.
4. `scripts/teob_fd_cache_controls.py --case catalog --public-td-npz docs/reviews/2026-09-15-teob-fd-attribution/catalog/public-td.npz --public-td-json docs/reviews/2026-09-15-teob-fd-attribution/catalog/public-td.json` checks conditioning, frequency bands, epoch counterfactuals, and paired public TD. The saved `catalog/generate_public_td.py` regenerates the public TD arrays bitwise.
5. `scripts/teob_fd_tail_controls.py --case bbh_reference` varies auxiliary spacing with fixed versus legacy support.

Public LAL is used only in diagnostics. Runtime remains native Torch.
Earlier reports under `2026-09-13-parity-review` and
`2026-09-15-continuous-matching` remain frozen. Cache hashes, commands, versions,
and source hashes accompany the new reports.

The bounded source change is preserved in [implementation.patch](implementation.patch).
