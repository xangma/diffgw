# HM London exact-N=61 diagnostic with independent D sampling

`report.json` records a bounded waveform diagnostic for the clean London HM
assembly after the Eq. (14) carrier contract correction.  The report uses
`evaluate_d_qnm_220(final_spin)` for the D carrier's absolute cyclic QNM
values, scales them by `total_mass/final_mass`, and keeps the N=61 London QNM
fits for the HM outer map.

The diagnostic installs a temporary explicit adapter over
`tests.parity_harness.get_fd_waveform` for aligned-spin `IMRPhenomHM` calls.
The adapter is restored before exit and does not change production dispatch.
The original catalog and holdout HM rows are selected only after validating
their frozen inventories; both are run with the unchanged strict gate
parameters.  Four fixed cases from `diagnose_hm_qnm_leaver_clean.CASES` are
also run as raw independent comparisons, with complete plus/cross, named PSD,
and H1/L1/V1 projection metrics retained in the report.  Runtime gradient
checks cover the independent D sampler and the corrected clean assembly.

The N=61 grid remains a metadata hypothesis inferred from published London
intervals.  This diagnostic does not certify inherited coefficient provenance,
fit coefficients to waveform residuals, or authorize production routing.

Run with:

```text
PYTHONPATH=src:tests:scripts:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
scripts/diagnose_hm_london_exact61_d_sampling.py \
--log-path docs/reviews/2026-09-13-parity-review/verification/hm-london-exact61-d-sampling/run.log
```
