# Independent TEOB QNM recalibration

Date: 2026-09-15. This is an experimental positive-spin (`0 ≤ χ ≤ 0.95`)
calibration, not a claim of exact release-fit recovery or whole-repository MIT
certification. No third-party code or calibration dataset was added; the
original generated coefficients are saved in this experimental report.

## Frozen method

The fit uses the published Eq. 91 basis
`Y=Y0(1+b1χ+b2χ²+b3χ³)/(1+c1χ+c2χ²+c3χ³)`, with `Y0` fixed at `χ=0`,
on 17 deterministic training spins and 7 disjoint direct-Leaver holdouts.
Targets are dimensionless `(2,2,n)` QNM frequency/damping quantities and
`alpha21 = alpha221 - alpha220`. Each quantity uses unweighted least squares;
negative spin is unsupported. The frozen dataset digest is
`880e3e65548811e38b589b22cb7ac7d45521c72a6669a2dbe520b45b0820e450` and the
calibration digest is `db18c0fdde1bdfe17e115d1ea9a54915a89b3023404571d8bec7ea30fe3a6283`;
see [refit-calibration.json](refit-calibration.json) and
[freeze-verification.json](freeze-verification.json).

The independent refit reduced held-out QNM RMSE relative to the rounded
published coefficients by factors **90.91 (ω220), 54.18 (α220), and 12.25
(α21)**. These are calibration-target errors, not waveform parity metrics.
On the exact 2,009 bins for the 30+20 nonspinning case (20–1024 Hz,
`df=0.5`, both polarizations), the baseline combined FD relative L2 is
`0.00032708438016984477`; the refit is
`0.0003298344651847134` (+0.8407876321%) and direct is
`0.00033039250070446134` (+1.011396672%). Refit-vs-direct residual norm over
the public norm is `3.4656130901691793e-6`. Fresh public arrays are bitwise
equal to frozen arrays, axes are exact, sources/artifacts are stable, and the
patched symbol was restored after each run. See [root verification](root-verification.json).

## Reproduction

From the repository root, run the clean refit and replay providers:

```sh
env PYTHONPATH=src:tests:scripts:. /Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
  scripts/refit_teob_qnm_independent_clean.py --points 17 \
  --output /tmp/torchwave-teob-qnm-refit-reproduction.json
env PYTHONPATH=src:tests:scripts:. /Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
  scripts/replay_teob_qnm_refit_clean.py --qnm-provider refit
env PYTHONPATH=src:tests:scripts:. /Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
  scripts/replay_teob_qnm_refit_clean.py --qnm-provider direct
```

Root reports 42 refit/assembly/conditioning/MIT-packaging tests plus 18 replay
tests (60 disjoint tests), recorded in [focused-test-results.json](focused-test-results.json).
Existing acceptance counts remain **28/34**; no canonical promotion or strict
rerun occurred.

The newly fitted coefficients were frozen before waveform validation. The
original paper coefficients were comparison-only and were never used to
optimize this refit. Independent fits are physically feasible and
have smaller calibration holdout errors, but this isolated substitution does
not close waveform parity. A compatible full postmerger calibration would need
physical NR training; the residual does not uniquely identify a causal source.
This does not certify `122022` or any other existing release, and does not
promote the default.
