# Pass 36: clean-room evidence boundary

Date: 2026-09-13

Full LAL parity is not achieved. Fresh strict verification passes **11/21**
catalog cases and **4/13** frozen holdout cases. The result arrays are
byte-for-byte identical to Pass 34, both inventories are valid, and neither
artifact reports manifest drift. The retained JSON and logs are under
[`verification/pass-35-final`](verification/pass-35-final).

Two independent Luna audits found no remaining runtime correction fully
determined by the allowed public primary evidence. The
[next-gap audit](pass-36-next-gap.md) rechecked all remaining model families
and passed 112 focused tests. The [gate inventory](pass-36-gate-inventory.md)
classifies all 19 nonpassing strict cases: nine numerical catalog failures,
one catalog case without an independent complete waveform oracle, and nine
holdout failures.

The remaining evidence requirements are:

- full-precision PhenomD and NSBH calibration and interpolation data;
- explicit HM/XHM calibration preprocessing, phase/time gauge, and mode-mixing
  conventions;
- exact precession version, frame, remnant, reference-phase, and degenerate-row
  conventions for Pv2, XP, and XPHM;
- a complete independent KerrRingdown waveform reference with excitation and
  normalization conventions; and
- versioned calibrated dynamics, NQC, merger, ringdown, and surrogate data for
  SEOBNRv4, SEOBNRv4_ROM, and TEOBResumS.

One local metadata defect was corrected: stored HM, NSBH, and TEOBResumS
validation maxima were stale relative to the final strict artifacts. The
manifest is synchronized and a focused regression test now derives the maxima
from the retained artifacts. The complete focused harness/parity run passes
64 tests. The last complete ordinary-suite run recorded **1,248 passed with 9
skipped** across eight successful shards; the newly added manifest regression
also passes in the current tree.

No LAL, PyCBC, TEOBResumS, or other waveform implementation source was read or
copied. Public LAL APIs were used only for black-box measurement. No constant
was fitted to an oracle residual, no gate was relaxed, and the deferred NR run
was not started. Further numerical movement without the evidence listed above
would require guessing or fitting to LAL output and would violate the stated
clean-room constraint.
