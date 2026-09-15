# NSBH composition diagnostic

`scripts/nsbh_composition_diagnostic.py` performs a black-box comparison of
TorchWave and public LALSimulation outputs for the catalog and holdout inputs.
It evaluates NSBH and PhenomD at `lambda2=0`, restores the specified NSBH
lambda, and sweeps `df={0.25,0.5,1}`, `f_final={512,1024}`, and
`f_ref={f_low,27 or 40,60}`. PhenomD has no tidal parameter, so its restored
entry is the same public Lambda-zero carrier. The resulting JSON is
`docs/nsbh_composition_diagnostic.json` and includes exact inputs, API
signature, package versions, source hashes, and amplitude/complex/affine and
nonaffine phase metrics.

Run with:

```text
PYTHONPATH=src:tests /Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
  scripts/nsbh_composition_diagnostic.py docs/nsbh_composition_diagnostic.json
```

At `df=0.25`, `f_final=1024`, and the frozen references (catalog `30`, holdout
`40` Hz), cross-backend NSBH complex L2 is `2.9515412285e-5` (catalog) and
`1.7579662428e-3` (holdout). The corresponding affine slopes are
`-2.9434788787e-7` and `-1.9586469273e-5` rad/Hz; nonaffine RMS is
`3.1770964e-9` and `1.0096027e-5` rad. Lambda-zero carrier comparisons are
`2.9427781e-6` and `5.9280127e-6` complex L2, while D carrier comparisons are
`2.3876242e-7` and `7.7121634e-7`. The cross-backend tidal-ratio complex L2 is
`1.5246428e-4` and `1.0422911e-2`.

Grid changes do not remove the residual. For catalog `f_final=1024`, NSBH
cross-backend L2 changes from `2.95154e-5` at `df=.25` to `2.94280e-5` at
`.5` and `2.92537e-5` at `1`; holdout changes from `1.75797e-3` to
`1.75280e-3` and `1.74252e-3`. At `df=.25`, extending `f_final` from 512 to
1024 changes catalog L2 `8.52642e-6` to `2.95154e-5` and holdout
`3.51385e-4` to `1.75797e-3`. The JSON also reports all requested
NSBH/D-composition and tidal-ratio fields, with the frozen holdout
`f_low=30, f_ref=40` explicitly represented.

The `common_sample_grid_records` section tests actual shared frequency values.
Each spacing and endpoint is first evaluated on the complete public grid for
both backends; only then are identical samples selected. Changing `df` from
`.25` to `.5` at fixed `f_final=512` gives zero complex difference for every
native and binary path (phase residuals are at machine precision).

Changing `f_final` from 512 to 1024 at fixed `.25`, restricted to
`30<=f<512`, produces a large affine difference for NSBH in both backends.
For catalog lambda, maximum complex L2 over references is `0.8971625` native
and `0.8971653` binary; for holdout it is `0.8813511` native and `0.8815881`
binary. Phase slopes are about `-0.0235356` rad/Hz (catalog) and
`-0.0228255` native / `-0.0228400` binary (holdout), with nonaffine RMS below
`2e-13` rad. PhenomD has zero difference. The earlier native-zero result was
caused by sparse native evaluation and was not a valid grid-invariance test.
No correction is inferred.
These are decomposition observations, not fitted corrections or anchor
choices.
