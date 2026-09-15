# Pass 10: NNLO remnant and independent verification

GPT-5.6 Luna handled implementation and independent source/code review. The
primary agent checked source passages and diffs, tightened tests, and ran public
binary verification separately. Existing calibration tables remain in use;
the deferred NR simulation was not restarted.

XP/XPHM now include the in-plane spin contribution in the signed final spin
for NNLO versions 101--104. The same remnant reaches XAS/XHM QNM, transition,
ringdown and mode-mixing calculations; final mass keeps the aligned fit. MSA
and ODE remnant policies remain unresolved. The [implementation report](pass-10-nnlo-remnant.md)
documents the primary-paper extra-square inconsistency and the earlier author
equations supporting the derived prescription. No formula was selected or fit
using waveform residuals.

Final verification: **1,075 passed, eight skipped in 74.40 s**. Strict catalog
acceptance remains **11/21**, with frozen holdouts **4/13**. No threshold or
frozen case was changed. Source, tests, scripts and manifest (166 files) stayed
unchanged during verification; the later manifest edit only records results.

The separate explicit comparison supplies NNLO version 102 and remnant option
0 to the public LAL API. Its maximum polarization raw relative L2 errors are:

| Fixed case | Before | After |
| --- | ---: | ---: |
| XP catalog input | 0.0697472 | 0.0611688 |
| XPHM catalog input | 0.0819129 | 0.0771457 |
| XP holdout input | 0.326295 | 0.309587 |
| XPHM holdout input | 0.343787 | 0.327692 |

These errors improve by about 5--12%; detector raw L2 errors also improve.
All four still fail strict acceptance. This diagnostic inventory does not
replace the default-setting frozen cases. In those cases, XP's catalog error
improves, while XPHM and both precession holdouts worsen. Other cases' complete
polarization/projection metric dictionaries are unchanged from pass 9.

The first full attempt exposed seven old rotation tests that assumed an
unchanged aligned remnant, plus an import error in the supplemental verifier.
The rotation tests now compare against a carrier with the same derived remnant,
at the original strict tolerances. Independent equation tests cover the remnant
formula, while these comparisons isolate reference-frame closure for all modes.
A separate reviewer confirmed this interpretation. The repaired verifier selects
the harness beside its native source; repeating the frozen baseline gave an
identical report. The [initial attempt](verification/pass-10/initial-attempt/)
is retained alongside the final results.

Evidence: [run record](verification/pass-10/verification-run.json),
[catalog](verification/pass-10/catalog.json), [holdouts](verification/pass-10/holdout.json),
[before/after comparison](verification/pass-10/comparison.json),
[source audit](verification/pass-10/source-audit.json), and
[changes from pass 9](verification/pass-10/changes-from-pass9.patch).
The ordinary suite exited 0; all three strict comparison runs exited 1 because
parity gaps remain. All verification processes have finished.
