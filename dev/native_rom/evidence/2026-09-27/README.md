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

## Historical QNM public-source gate

The [frozen plan](qnm_historical_source_gate_plan.json), [public-source audit](qnm_historical_source_audit.json), [evaluator](../../diagnose_qnm_historical_source_gate.py), and [machine-readable result](qnm_historical_source_gate.json) test whether the pinned public sources completely define the historical `V4-QNM-INTERPOLATION` prescription. Bohé et al., Sec. II.C and Appendix B, identify interpolation of Berti et al. numerical results. Berti et al., Table II and Appendix E, provide eleven rounded numerical rows and separate approximate fits. The hash-pinned, unversioned Berti author archive supplies a dense 10,000-row grid, but none of these sources identifies the exact historical subset and retained precision, interpolation family/order, endpoint conditions, signed-domain mapping, or out-of-domain behavior.

The SHA-pinned audit ran on `len` and satisfied only one of six mandatory requirements. The five absent fields are recorded as blockers, so the source gate fails and `V4-QNM-INTERPOLATION` remains pending. Cook's 2019 data remain an independent numerical validation source, not a definition of a 2017 historical implementation. This audit did not inspect LAL source, query a binary oracle, transcribe a historical table, retune a fit or threshold, change production code, or qualify release.

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

## Published-flux initial-condition domain gate

The [prospectively frozen plan](ic_paper_flux_domain_plan.json),
[tests](../../reference/tests/test_initial_conditions.py), and
[machine-readable result](ic_paper_flux_domain.json) extend the conditional
`V4-IC-PAPER-FLUX` comparison from two points to eight rows spanning `q=1–8`,
`|chi_i|≤0.85`, and `M Omega=0.015–0.045`. Both frontends compare `r`,
`p_r*`, `p_phi`, and the 35-mode pre-NQC torque with the 50-digit scalar
composition at the unchanged `3e-10` relative / `2e-12` absolute gates. Each
scalar row is also stable at 70 digits to `1e-35` relative. All 26 tests passed
on host `len` in 89.75 s; the result records the CPU-only command, environment,
cwd, and completed log. An earlier environment rejected JAX during import due
to an old-JAX/SciPy API mismatch and is recorded separately from physics.

No production source, threshold, catalog, or release status changed. The grid
does not identify the historical flux switches or Cartesian root procedure and
does not validate integration, events, or convergence. `V4-ODE-IC` and
`V4-FLUX-4-9` therefore remain pending.

## Published Pan-rho43 discriminator

The [prospectively frozen plan](flux_43_pan_rho_plan.json),
[focused tests](../../reference/tests/test_flux_43_branch.py), and
[machine-readable result](flux_43_pan_rho.json) add an explicit independent
scalar branch for Pan Eq. (29g)'s higher-spin terms inside `rho43**4`. The
existing Taylor-f placement and the printed Pan-rho placement both agree with
Torch and JAX at the unchanged `4e-12` relative / `2e-14` absolute gates on
three frozen supplied states. The strong row separates the branches by
`0.15113433306957774`; zero-spin and equal-mass unequal-spin controls coincide,
and each scalar branch is stable from 50 to 70 digits at `1e-35` relative.

The CPU-only run on `len` passed all 16 checks in 5.84 s; the bounded published,
focused, and frontend regression passed 60 tests in 14.45 s. This removes
formula-level dependence on implementation source for this discriminator, but
the papers do not select the exact historical branch. No production code,
release threshold, catalog entry, or release status changed; `V4-FLUX-4-9`
remains pending.

## Historical-fidelity release boundary

The [current-head gradient plan](jax_pathwise_whole_gradient_current_plan.json)
replays the mandatory fixed-mesh whole-mode gradient gate on `len` with JAX
and JAXlib 0.11.2. Its [result](jax_pathwise_whole_gradient_current.json)
passes all four unchanged checks: eager/JIT whole-mode relative L2 is
`3.437010236069393e-11` against `1e-10`, the largest AD/FD relative L2 is
`1.711951089644063e-4` against `2e-3`, the gradient L2 is
`2.292175317675213`, and replay state error is `5.195843755245733e-14`.
This closes the frozen gradient gate only; accepted meshes, event decisions,
physical parameters, and production registration remain outside its scope.

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
