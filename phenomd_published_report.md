# Published PhenomD fit transcription

Source: S. Khan et al., “Frequency-domain gravitational waves from nonprecessing black-hole binaries. II. A phenomenological model for the advanced detector era,” arXiv:1508.07253v2, Eq. (31) and Table V. The permitted primary LaTeX source was `/tmp/torchwave-PhenomPaper2.tex`, SHA-256 `ed0d9ca1031def94a8df3d2964967dd88d50aa179bd12fd2bbf1cfb067070b44`.

`eval_table_v_fit` evaluates the printed 11-term basis: constant and linear `eta`, then powers `(chi_pn - 1)^1`, `^2`, and `^3`, each with constant, linear, and quadratic `eta` terms. The helper literal Eq. 31 uses the paper’s explicitly stated unscaled reduced-spin coordinate, while the paper’s earlier normalization of the same symbol creates an ambiguity; the diagnostic covers both interpretations.

`chi_pn = chi_eff - (38 eta / 113) (chi_1 + chi_2)`,

where `chi_eff = (m_1 chi_1 + m_2 chi_2)/M`. The alternate normalized coordinate is `hat_chi = chi_pn / (1 - 76 eta / 113)`.

The implementation preserves the displayed coefficient literals and evaluates in the broadcast input dtype/device with PyTorch autograd support. Independent tests cover all 19 row names/order, selected literal precision, the expansion origin and eta dependence, complete polynomial basis, broadcasting, and gradients.

Validation command:

`PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q tests/test_phenomd_published_fits.py`

Result: 3 passed.

## Black-box diagnostic

The native runtime evaluator signature obtained with `inspect.signature` was
`IMRPhenomD.eval_table_v(name: str, eta: Tensor, chi_pn: Tensor) -> Tensor`.
The diagnostic patched that class attribute, counted 72 replacement calls per
coordinate run (18 per case), and restored the original callable descriptor in
`finally` (`restored=true`).
The four inputs were `IMRPhenomD_diagnostic` plus holdouts 0, 1, and 2 from the
current-gates catalog. The frozen gates were complex relative L2 <= 1e-6 and
mismatch <= 1e-8.

Baseline hp/hc complex L2 (identical to displayed precision for both
polarizations) was respectively `4.947651637e-7`, `1.232190226e-5`,
`8.962919937e-7`, and `9.629342181e-7`; magnitude L2 was
`7.538818901e-8`, `1.559841362e-6`, `1.400517815e-7`, and `2.845477577e-7`.
Under the Eq. 31 unscaled interpretation, hp/hc complex L2 became
`4.565850611e-3`, `3.368545060e-3`, `5.229587713e-3`, and `1.109829417e-2`;
magnitude L2 became `5.172974553e-5`, `1.012538550e-5`, `8.654273523e-5`, and
`1.752486395e-4`. Under the normalized interpretation (`chi_pn` divided by
`1-76 eta/113` before Eq. 31), they became complex L2
`6.106691602e-1`, `1.062311937`, `6.924476223e-1`, and `1.273989963`, with
magnitude L2 `1.440745195e-3`, `1.105837045e-2`, `2.148495163e-3`, and
`7.712715493e-3`. Detector H1/L1/V1 raw and magnitude errors are retained in
the diagnostic run output; the unscaled first-case H1 values are
`0.004565969225` and `5.173064500e-5`.

These scores are diagnostics only. The paper itself is ambiguous: its prose
and Eq. 31 explicitly call the fit coordinate unscaled `chi_PN`, while the
nearby definition normalizes the same symbol to `hat{chi}`. No oracle score was
used to select an interpretation.

The Table-V transcription was compared programmatically against all 19 primary
TeX rows and 11 columns after scientific-notation normalization; all 209
values matched. Source hash remained
`ed0d9ca1031def94a8df3d2964967dd88d50aa179bd12fd2bbf1cfb067070b44` before and
after. New module, diagnostic, and test hashes after the run are
`55992c5c6d46e1430d2c86915dcedf44741beaddf039ff95faea944531bde2dd`,
`211991a869a20c3fb9a9f149b8a90713aee1e1d47cf7d7f0c3540664e874e799`, and
`6a6c855239b8e5b414524f932e7b21516f6581d2e50ac819bd84b0667d3be30a`. The
structured JSON also hashes the native runtime modules and the catalog/holdout
input files before and after; all were unchanged.
