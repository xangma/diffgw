# TEOB PA circular initialization clean diagnostic

Status: diagnostic-only; the canonical production PA route was not changed. The reviewed runtime also exposes the direct policy as an opt-in helper for labeled diagnostics.

## Contract and derivation

For `x=j^2`, `U=A/r_c^2`, and `H_eff^circ=sqrt(A+U*x)+G*j`, direct squaring of the radial stationarity equation gives `a*x^2+b*x+c=0` with `a=U'^2-4*U*G'^2`, `b=2*A'*U'-4*A*G'^2`, and `c=A'^2`. The zero-spin probe has `r_c=r`, `r_c'=1`, and `G'=0`; it evaluates `U'=r_c^-2*(A'-2*A*r_c'/r_c)` directly. A negative discriminant is replaced by zero only when negative. At exact `G'=0`, the lower root is selected, as required by the compatibility contract. No perturbation or fitted value is used.

Primary equation sources: [arXiv:1805.03891v2 Sec. II](https://arxiv.org/html/1805.03891v2) and [arXiv:1806.01772v2 Appendix C, Eq. (58)](https://arxiv.org/html/1806.01772v2).

The local shadow reruns the clean alternating PA equations with leading-order zero-spin horizon absorption. Its only intended change is the initial circular `j` profile; the actual clean helper is run separately as the baseline and an analytic shadow is used as a reproduction control. The shadow calls the opt-in `circular_angular_momentum_squared` helper; the canonical analytic initializer remains the default.

## Frozen probe

The grid is `r0=19.999999999999996`, `r_min=14.0`, `dr=0.1`, `nu=0.24`, zero spin/tide, and PA orders `[2, 4, 6, 8]`. The exact float64 floor policy produces 60 nodes and ends at `r=14.099999999999996`; `r_min` is the lower bound passed to the clean runtime. The supplied public-binary endpoint is recorded for validation only: `{"j": 4.215698565112572, "omega": 0.018973840039422412, "phase": 132.17031461096497, "pr_star": -0.0012978461179493459, "time_M": 9523.942662602482}`. It is not used as an input or fit target.

## Circular equation diagnostics

In float64 there are 18 positive, 12 negative, and 30 exactly zero raw discriminants over 60 nodes. The raw discriminant range is -2.5849394142282115e-26 to 2.5849394142282115e-26; its maximum magnitude relative to `|b^2|+|4ac|` is 1.8840424346519066e-16. The lower root is positive and finite, and its maximum absolute difference from the clean analytic `j^2` is 4.3074917144281244e-07 (maximum `j` difference 4.5720650376779304e-08).

The direct stationarity checks remain finite: maximum absolute polynomial residual 2.710505431213761e-20 and maximum absolute unsquared residual 8.430785289129531e-11. The float32 control has 14 positive, 21 negative, and 25 zero raw discriminants, exposing the expected larger representation sensitivity without changing the branch rule.

## PA endpoint comparison

Each row below reports the clean production result versus the local quadratic-initializer shadow. `production_control_reproduction` checks that the shadow mechanics reproduce the production helper when seeded with the current analytic initializer; `general_quadratic_shadow` is the requested Eq. (58) comparison.

| PA order | clean time M | quadratic time M | time change M | clean phase | quadratic phase | phase change | clean j | quadratic j | clean p_r* | quadratic p_r* |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 9521.62015108568 | 9521.61975256764 | -0.000398518 | 132.135043639474 | 132.13503871051 | -4.92896e-06 | 4.21569808208875 | 4.21569824338483 | -0.00129883495244751 | -0.00129882698834732 |
| 4 | 9523.94801799516 | 9523.9486632954 | 0.0006453 | 132.170401410245 | 132.170398998739 | -2.41151e-06 | 4.21569858451862 | 4.21569846308668 | -0.00129784299963585 | -0.00129787269083923 |
| 6 | 9523.94212328442 | 9523.94269715737 | 0.000573873 | 132.170304518283 | 132.170318245488 | 1.37272e-05 | 4.21569858060851 | 4.21569851099511 | -0.0012978485463599 | -0.0012978507756978 |
| 8 | 9523.94216423246 | 9523.94270531117 | 0.000541079 | 132.170305220601 | 132.170315504463 | 1.02839e-05 | 4.21569858067726 | 4.21569854705764 | -0.00129784849230281 | -0.00129784554355592 |

The analytic control reproduction maximum profile differences are listed in the JSON artifact. Any nonzero control difference is an implementation-copy check, not a parity result. The binary endpoint remains validation-only; no alignment or tuning was performed.

## Interpretation

At order 8 the general shadow changes the clean endpoint by `0.00054107870892 M` in time and `1.02838623945e-05 rad` in phase, while the validation-only public endpoint is offset from clean by `0.000498370021887 M` and `9.39036422665e-06 rad`. Thus this independently derived general initializer accounts for most of the observed endpoint displacement in this frozen probe, but leaves a residual and does not establish production parity. The analytic initializer remains the canonical production default; the general policy is exposed only as an opt-in helper for labeled diagnostics pending coordinator review.

## Reproducibility

Host `RONI2`, cwd `/Users/xangma/repos/torchwave`, Python `/Users/xangma/miniconda3/envs/pycbcorig313/bin/python`, command `scripts/diagnose_teob_pa_circular_initialization_clean.py --output docs/reviews/2026-09-13-parity-review/verification/teob-pa-circular-initialization-clean/report.json`. Runtime source hashes before/after and the complete machine-readable metrics are in `report.json`; `runtime_sources_unchanged=True`.

No LAL/GPL implementation source, reference cache, source-only calibration table, public waveform, or oracle fitting was read by this diagnostic.
