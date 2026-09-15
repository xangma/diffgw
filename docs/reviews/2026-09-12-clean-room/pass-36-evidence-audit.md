# Pass 36: evidence and manifest audit

Date: 2026-09-13

The Pass 34 and Pass 35 summaries are internally consistent with the retained
strict artifacts. `verification/pass-34-final/catalog.json` reports 21 cases
and 11 passes; `holdout.json` reports 13 cases and 4 passes. Their per-case
statuses and maximum polarization/projection L2 values agree with the Pass 34
tables. The eight-mode TaylorF2_HM inventory, the PhenomHM `(2,1)` localization,
and the NSBH residuals in Pass 35 agree with their detailed reports. The
summaries state the clean-room boundary consistently: no implementation source
was read or copied, no oracle residual fitting was used, and no NR run was
started.

One metadata error was found and corrected in `docs/model_manifest.json`: the
current top-level verification record pointed to the older Pass 21 report and
reported 1,161 tests on four shards. The retained Pass 33 full-suite artifacts
prove 1,248 passed and 9 skipped across eight successful shards; the slowest
shard took 71.42 seconds. The record now points to Pass 34 and the Pass 33
artifact directory, with those counts and timing.

A subsequent per-model comparison against the fresh Pass 35 artifacts found
stale validation maxima for HM, NSBH, and TEOBResumS. Those values and the new
regression coverage are recorded in the
[Pass 36 gate inventory](pass-36-gate-inventory.md). JSON parsing and
`git diff --check` passed. Runtime code was not inspected or changed.
