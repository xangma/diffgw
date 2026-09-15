# Independent v3 search audit

**Audit date:** 2026-09-15  
**Scope:** read-only identity, policy, screening, split, freeze, and post-freeze validation checks. No waveform downloads, fitting, LAL calls, or source changes were performed by this audit.

## Findings

- The pinned policy is unchanged: `quality-policy.json` SHA-256
  `1a863811b90c410cd5a9b0b52a0bd53696b869a0d503cb8fa61d9c054842138d`.
- The merged input manifest has SHA-256
  `7689bd20535827278c240ae581ad70b6f2bf2b808fc7517b24a426abb674f7b5` and records both source manifests. Its v2 source hash matches the file on disk (`3ee69110faf64800e253e9124e736e9c5a16aad53e40bbd4ccd57fe5e214ccaa`), and all 20 retained v2 rows for q=1.5, 2, 3, 4, 5 are exact dictionary matches, including metadata, paths, source identities, and NPZ hashes.
- The new source manifest hash is `ce4680f3860ba05e07b007d9a5b5750251fc830f36870b5459bd93917497a64c`. Selection preserves the predeclared order: q6 `SXS:BBH:2489` rank 2 (holdout), with `SXS:BBH:2164` rank 3 retained as the next candidate; q8 `SXS:BBH:2707` rank 2 (training). Historical failed rank-1 candidates remain recorded outside the new search.
- The fixed quality screen passes all seven targets (`overall_pass: true`): five training and two holdout primaries. Selected primaries are exactly the qualified-manifest rows, with q3/q6 holdout and q1.5/q2/q4/q5/q8 training. The qualified manifest SHA-256 is `990df8e246ba978fc47735407be05b49eb9a3f261565ccc4a946029704cc5160`; its screening block points to the merged-manifest hash and pinned policy.
- New candidate gate evidence passes the fixed limits: q6 post-peak duration 154.0002 M, extraction 0.0020457, next-resolution 0.0026633; q8 post-peak duration 150.1935 M, extraction 0.0001303, next-resolution 0.0008023. All spin, eccentricity, preprocessing, and split gates pass. These are NR-only admission values.
- Calibration freeze is internally consistent. Calibration SHA-256 is `ade377c62a57a5d9cff513e34f22b0fe43677177a0c1577a9a4af5ad3b8aa561`; split and fit provenance point to the qualified manifest and quality report. Training contains q=1.5, 2, 4, 5, 8; holdout contains q=3, 6.
- Frozen validation is post-freeze and amplitude-only. Prediction and target artifacts are frozen, use the same two holdout IDs, exact frequency grids, and unchanged calibration digest before/after. Reported fit-vs-NR relative amplitude L2 is 0.0065326 (q3) and 0.0079741 (q6); the post-freeze LAL report records fit-vs-LAL 0.0090555 and 0.0050111, and NR-vs-LAL 0.0050245 and 0.0077430, respectively.

## Evidence files

- `nr33-manifest-round1.json`
- `nr33-new-manifest-round1.json`
- `selection-round1.json`, `preflight-verification.json`
- `quality-screen-round1.json`, `nr33-qualified-manifest.json`
- `nr33-calibration.json`, `nr33-calibration-split.json`, `nr33-calibration-freeze.json`
- `validation/nr33-validation.json`, `validation/nr33-frozen-predictions.json`, `validation/nr33-heldout-targets.json`

The audit checks hashes and JSON relationships directly; it does not establish independence of the replacement simulations from the broader SXS numerical family, and the validation does not evaluate complex-waveform mismatch.
