# Pass 13: XHM normalization, fallback and phase-node corrections

Completed 13 September 2026 with two fresh-context **GPT-5.6 Luna** subagents
and parent integration/verification. **Full LAL parity remains unfinished.**
The strict catalog passes **11/21** and frozen holdouts **4/13**.
No new strict case passes, and all four full XHM/XPHM complex errors worsen.
This pass corrects source-defined behavior without demonstrating full-waveform
parity progress.

## Retained changes

- Intermediate amplitude fit targets use the paper's frequency-independent
  `A0`: inverse-polynomial inner values are `1/(A/A0)`. Conversion to the
  native frequency-dependent leading-amplitude ratio occurs once at evaluation.
- Parameter vetoes remove inner values and retain endpoint derivatives.
  Denominator failure instead removes endpoint derivatives and retains four
  values. Each fallback selects one polynomial for the complete interval;
  a failing cubic selects the endpoint linear reconstruction globally.
- Nonmixing phase reconstruction uses the published default nodes
  `[1,2,3,4,6]` and their inherited fit tables, replacing `[1,2,3,4,5]`.

The [patch](verification/pass-13/changes-from-pass12.patch) changes one runtime
file, two new test files and two existing tests. Earlier dirty work is
preserved. Calibration tables, acceptance inventories, grids and thresholds
are unchanged. See the [amplitude derivation](pass-13-amplitude.md),
[phase review](pass-13-phase-metadata.md),
[independent fallback review](verification/pass-13/fallback-independent-review.md)
and [integration review](verification/pass-13/integration-review.json).

## Verification

| Check | Result |
| --- | --- |
| Full ordinary suite | **1,120 passed, 8 skipped, 76.18 seconds** |
| Frozen strict catalog | **11/21 passed** |
| Frozen strict holdouts | **4/13 passed** |
| Separate explicit component checks | **3/34 passed** |
| Mode-sum closure controls | **4/4 passed** |

The catalog still contains nine numerical failures and one unresolved complete
reference. Component successes overlap carrier checks and add no frozen parity
success. Ordinary tests establish mathematical/API behavior, not LAL agreement.
Six skips require CUDA and two require generation-only gmpy2; MPS tests pass.

Changed frozen maximum polarization relative L2 errors (before → after):

| Case | Complex strain | Magnitude only |
| --- | ---: | ---: |
| IMRPhenomXHM_diagnostic | 0.10095288 → 0.32766557 | 0.058345604 → 0.25884171 |
| IMRPhenomXPHM_diagnostic | 0.088428822 → 0.10160197 | 0.065965035 → 0.065554273 |
| IMRPhenomXHM_holdout_6 | 0.23418847 → 0.65967621 | 0.12278694 → 0.44463141 |
| IMRPhenomXPHM_holdout | 0.42243628 → 0.56006875 | 0.32719381 → 0.44846581 |

All four changed full-waveform complex errors worsen. Some isolated component
amplitude errors improve; this does not establish full-waveform progress.
All other frozen case metrics are unchanged. The complete measurements and
unchanged-inventory assertions are in the
[comparison](verification/pass-13/comparison.json). Improvements and regressions
are reported together; no residuals were supplied to the implementers or used
to fit coefficients. The earlier normalization/node-only candidate is preserved
under [normalization-and-nodes](verification/pass-13/normalization-and-nodes/README.md).
Subsequent source review identified the distinct Table VI fallback rule and
both Luna reviews confirmed it before final verification.

The first fallback verification found two older unconditional C1 amplitude
assertions. The published pole branch removes those derivative conditions.
Its default test now proves the pole branch is active and preserves C0 and
phase checks. A separate complete-waveform control sets inner fit values from
the actual endpoint Hermite polynomial, proves the production quintic clears
the detector, and verifies C0/C1 at both amplitude joins. No tolerances were
relaxed. The intermediate failed run is preserved under
[fallback-before-join-test-correction](verification/pass-13/fallback-before-join-test-correction/README.md).
The final focused integration passed **57 tests**.

The final verifier ran on **RONI2**, cwd `/Users/xangma/repos/torchwave`, driver PID
**21015**, from `2026-09-13T10:06:41.072031+00:00` to `2026-09-13T10:08:16.493010+00:00` using
`/Users/xangma/miniconda3/envs/pycbcorig313/bin/python` and the
[driver](verification/pass-13/verify.py). All children finished; exits are
0/1/1/1 for ordinary/catalog/holdout/component checks.
Exact commands, child PIDs and logs are in the
[run record](verification/pass-13/verification-run.json). All **173**
runtime/test/script/manifest hashes remained stable during verification.
The later manifest edit records results separately. `git diff --check` passes.

## Remaining limits

The pole detector still uses the inherited 11-point/5%-of-endpoint heuristic;
this is not an exact root search. Ordered absolute-amplitude vetoes remain
incomplete. Sparse-calibration phase-node selection, independent p6 table
provenance and author phase-fit preprocessing remain unresolved. No speculative
relative phase-target conversion was introduced. These gaps and the existing
MSA, EOB, ROM and Kerr waveform work prevent a credible completion percentage.
See the [remaining-work assessment](pass-12-remaining-work.md).

Implementers used primary papers, author mathematical assets and inherited
native code. LAL/PyCBC implementation source and binary residuals were excluded;
parent public-binary verification was separate. This instruction boundary does
not certify inherited provenance or provide OS-enforced isolation. Existing
tables remain in use; the deferred NR job was not restarted or scheduled.
No commit or pull request was created.
