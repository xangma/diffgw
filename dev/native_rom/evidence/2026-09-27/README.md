# Finite-point certificate: `V4-QNM-NATIVE-220-SPLINE`

This receipt checks the existing diffGW signed-spin, degree-five B-spline QNM rule at 12 declared interstitial spins: `±0.5005`, `±0.9005`, `±0.9905`, `±0.99971875`, `±0.999859375`, and `±0.9998875`. The [frozen plan](qnm_spline_component_plan.json), [diagnostic](../../diagnose_qnm_spline_component.py), and [machine-readable result](qnm_spline_component.json) record the inputs, checks, and every finite-point value. No coefficients or production code changed.

The independent numerical reference is Gregory B. Cook, *Kerr Quasinormal Modes: s=-2, n=0--7*, [Zenodo record 2650358](https://zenodo.org/records/2650358), Version 1 (24 April 2019), DOI [10.5281/zenodo.2650358](https://doi.org/10.5281/zenodo.2650358). The [record API](https://zenodo.org/api/records/2650358) identifies the license as `odc-by`, the [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/). The temporary validation input was `KerrQNM_00.h5`, SHA-256 `d5853acabc1a07b66b4fae437c02e1435a9d2b3561bd7e4752b9721c9cd48923` (record MD5 `34c04f3b03bbb99935d1f91f7296bd02`), datasets `/n00/m+02/{2,2,0}` and `/n00/m-02/{2,-2,0}`. The archive is not committed; the result contains only derived finite-point values and errors, with this attribution.

At each spin, degree-seven and degree-nine local polynomials fitted to the nearest eight and ten Cook rows in `x=-log(1-|j|)` give independent value and signed-spin derivative estimates. Their derivative difference checks stencil sensitivity. The native spline derivative is analytic; Torch autograd and JAX `jacfwd` independently check the two frontend derivatives. The author-derived derivative and stencil checks are empirical numerical estimates, **not** a rigorous interpolation remainder bound.

| Frozen finite-point check | Maximum observed | Gate |
| --- | ---: | ---: |
| Spline value versus ten-row Cook estimate, absolute | `9.62e-13` | `<1e-8` |
| Analytic derivative versus ten-row Cook estimate, scaled by `max(1, |author derivative|)` | `1.28e-9` | `<1e-4` |
| Eight- versus ten-row Cook derivative estimates, same scale | `3.03e-9` | `<1e-4` |
| Torch versus JAX derivative, absolute | `4.13e-11` | `<1e-8` |
| Analytic versus either frontend derivative, absolute | `1.02e-10` | `<1e-8` |

All six frozen gates passed. The sixth reuses the SHA-pinned [current-source fixed-boundary waveform receipt](../2026-09-25/qnm_ringdown_waveform_current.json): replacing only QNM for its `q8_high_aligned` row changes the whole-mode complex relative L2 by `2.5494566848309676e-6`, below the predeclared `1e-5` gate; baseline reconstruction over that fixed boundary is exactly zero. This is one waveform-impact witness, not a domain-wide waveform result.

The diagnostic ran on `len` under the pinned native source and prior-receipt hashes in the plan. It completed in 4.12 s with `status=diagnostic_complete`; its JSON preserves the 12 rows and pass/fail gates. This is a **finite-point component certificate**. It establishes neither a continuous signed-spin error bound nor historical natural-cubic/binary parity, and it does not qualify the full SEOBNRv4 generator or authorize production promotion.

## JAX whole-path blocker

The [frozen directional-amplification plan](jax_nqc_directional_amplification_plan.json),
[diagnostic](../../diagnose_jax_nqc_directional_amplification.py), and
[result](jax_nqc_directional_amplification.json) test the remaining eager/JIT
gap without changing production code or the `1e-10` gate. On one identical
fixed chart, a `1.0500e-15` relative perturbation of the pre-NQC mode produces
a `4.2876e-10` whole-mode change, an amplification of `4.0834e5`. This
reproduces a gate failure, but it is only `27.7%` of the frozen
`1.547138e-9` pathwise gap and the half-step response is nonlinear. The exact
blocker is therefore narrower than a complete causal claim: no gate-safe
repair is identified, and both the unresolved pathwise remainder and exact
historical numerical semantics remain to be qualified. The gradient gate
stays failed.

The bounded CPU rerun used JAX 0.11.2 on host `len`, cwd
`/home/xangma/tmp/diffgw-seobnr-jax-amplification-20260927`, Python PID
`2190365` (launcher PID `2190360`), and log
`jax_nqc_directional_amplification-rerun.log`. It completed in 49.5 s with
the expected diagnostic exit status 1 because two predeclared attribution
checks failed; no process or next check remains.

## Historical-fidelity release boundary

The [frozen release plan](historical_fidelity_release_plan.json),
[evaluator](../../diagnose_historical_fidelity_release.py), and
[machine-readable result](historical_fidelity_release.json) turn the selected
historical-fidelity criterion into one machine-readable boundary. Source,
whole-waveform, numerical, gradient, and independent-ROM holdout gates are all
mandatory. The measured `SXS:BBH:2569` native/public overlaps below `0.99` are
preserved as a physical-accuracy disclosure, not relabeled as a historical-
fidelity pass or used to tune the historical model. A successful evaluator run
means the exclusion boundary is enforced while the candidate remains
unqualified; it does not mean the model passed release qualification.
