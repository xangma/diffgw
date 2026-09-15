# Pass 35: remaining bounded clean-room audits

Date: 2026-09-13

Pass 35 found no further runtime change justified by the allowed primary
sources. It closes three remaining bounded candidates:

- The public PhenomHM one-mode API localizes the largest non-`(3,2)` residual
  to `(2,1)`, with optimized complex L2 `0.150` in the diagnostic and `0.167`
  in the holdout. `(3,3)` and `(4,4)` are at the percent level. The published
  mapping does not supply a missing unique coefficient or convention.
- The published TaylorF2_HM `(5,5)` amplitude yields a candidate leading
  ratio, but the primary evidence does not independently identify its
  source-frame mode with TorchWave's stored negative-`m` SPA phase convention.
  The supported inventory therefore remains the eight source-backed modes
  through `l=4` recorded in `docs/model_manifest.json`.
- Fresh NSBH branch probes and a primary-equation audit found no new
  amplitude, window, classification, or reference-frequency defect after the
  sentinel correction. The residual remains `2.965e-5` in the catalog and
  `1.758e-3` in the holdout. The remaining difference is consistent with the
  already documented affine phase-gauge and unavailable coefficient-precision
  evidence.

The detailed evidence is in
[PhenomHM localization](pass-35-hm-mode-localization.md),
[Taylor `(5,5)`](pass-35-taylorf2-hm-55.md), and
[NSBH blocker](pass-35-nsbh-blocker.md).

## Final strict verification

Fresh strict catalog and frozen-holdout runs after the manifest update remain
at **11/21** and **4/13** passing cases. Both inventories are valid, with zero
manifest-drift errors. Their result arrays are byte-for-byte identical to the
Pass 34 final artifacts. The JSON, logs, PIDs, and expected nonzero strict exit
codes are retained under
[`verification/pass-35-final`](verification/pass-35-final).

The last complete ordinary-suite run recorded **1,248 passed with 9 skipped**
across eight successful shards. The independent [Pass 36 next-gap audit](pass-36-next-gap.md)
found no further runtime correction fully determined by the allowed public
primary evidence; its 112 focused tests passed. The subsequent
[gate inventory](pass-36-gate-inventory.md) classifies every nonpassing case
and synchronizes stale manifest measurements to the final artifacts.

No LAL, PyCBC, TEOBResumS, or other waveform implementation source was read or
copied. No coefficient was fitted to public binary output, no strict gate was
changed, and no NR job was started.
