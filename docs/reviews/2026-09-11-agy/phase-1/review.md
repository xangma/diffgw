# Phase 1 Clean-Room Parity Review: IMRPhenomXAS Amplitude Derivative & Phasing Restoration

**Date**: September 11, 2026  
**Target**: `src/torchwave/approximants/imrphenom_xas.py`  
**Literature Reference**: Pratten et al. (2020), *Setting the cornerstone for the IMRPhenomX family of models for gravitational waves from compact binaries: The dominant harmonic for non-precessing quasi-circular black holes*, Phys. Rev. D 102, 064001 ([arXiv:2001.11412v1](https://arxiv.org/abs/2001.11412v1)).

---

## 1. Summary of Changes

### 1.1 Restoration of Published Phase Collocation Weighted Average (Eq. 95)
- **Defect**: An Agy attempt during this review incorrectly treated the already-correct pre-dispatch weighted average as an ad-hoc heuristic and removed it in favor of an unweighted fit difference.
- **Published Physics**: Section VII.B (Eq. 95) of Pratten et al. (arXiv:2001.11412v1) explicitly specifies:
  $$v_2^{\rm int} = 0.75 \left(v_4^{\rm RD} + \Delta v_2^{\rm RD,4}\right) + 0.25 v_2^{\rm direct}$$
  This weighted average is calibrated published physics designed to tame unphysical behavior at large mass ratios and spins.
- **Resolution**: Restored `v2_direct = -XASParameterFits.fit_int_phase_v2(eta, s1_eff, s2_eff)` and the canonical $0.75/0.25$ weighted average with an explicit literature citation to Eq. (95). No other phase formulas were modified.

### 1.2 Exact Analytic Inspiral Amplitude Frequency Derivative (Eqs. 68–69, 72, 77)
- **Defect**: The intermediate amplitude $C^1$ boundary match at $f_1 = f_{\rm trans, amp}$ used a forward finite difference with step $\epsilon = 10^{-6}$ (`df_eps`), introducing an unnecessary $O(10^{-6})$ truncation error into the Hermite collocation system and degrading $C^1$ matching.
- **Mathematical Derivation**:
  - The 3PN TaylorF2 amplitude expansion (Eq. 69) is a sum of fixed powers of $x = \pi M f$:
    $$\mathcal{A}_{\rm PN}(x) = t_1 + t_2 + t_3 + t_4 + t_5 + t_6$$
    with exponents $0, 2/3, 1, 4/3, 5/3, 2$. Its exact derivative with respect to $x$ is:
    $$\frac{d\mathcal{A}_{\rm PN}}{dx} = \frac{1}{x} \left( \frac{2}{3} t_2 + t_3 + \frac{4}{3} t_4 + \frac{5}{3} t_5 + 2 t_6 \right)$$
    By the chain rule, the derivative with respect to dimensionless frequency $M f$ is:
    $$\frac{d\mathcal{A}_{\rm PN}}{d(Mf)} = \pi \frac{d\mathcal{A}_{\rm PN}}{dx}$$
  - The pseudo-PN inspiral amplitude terms (Eq. 68) are $\sum_{i=1}^3 \rho_i x^{(6+i)/3}$. Their exact frequency derivative with respect to $Mf$ is:
    $$\frac{d\mathcal{A}_{\rm pseudo}}{d(Mf)} = \pi \left( \frac{7}{3} \rho_1 x^{4/3} + \frac{8}{3} \rho_2 x^{5/3} + 3 \rho_3 x^2 \right)$$
  - The intermediate amplitude ansatz models the inverse amplitude (Eq. 72):
    $$\frac{1}{\mathcal{A}_{\rm int}(f)} = \sum_{i=0}^5 \alpha_i (Mf)^i$$
    Enforcing $C^1$ continuity at the inspiral boundary yields the derivative condition (Eq. 77):
    $$\left. \frac{d}{d(Mf)} \left(\frac{1}{\mathcal{A}}\right) \right|_{f_1} = -\frac{\mathcal{A}'_{\rm ins}(f_1)}{\mathcal{A}_{\rm ins}(f_1)^2}$$
- **Resolution**: Extended `pn_amp_func` with keyword flags (`derivative`, `return_both`) to return the exact analytic derivative without duplicating coefficient tables. Replaced the forward difference in the collocation system with the exact analytic sum $d_1 = \mathcal{A}'_{\rm PN}(f_1) + \mathcal{A}'_{\rm pseudo}(f_1)$, preserving all parameter gradients under autograd and eliminating forward graph hooks.

---

## 2. Verification Suite

New test file: `tests/test_xas_amplitude_derivative.py`
- `test_pn_amp_exact_derivative_vs_complex_step`: Validates `pn_amp_func` analytic derivative against independent complex-step differentiation ($h = 10^{-25}$) across 5 spin/mass-ratio configurations to $< 10^{-11}$ relative error.
- `test_pn_amp_func_return_both`: Validates that combined evaluation matches separate calls identically.
- `test_xas_execution_contexts`: Validates exact bitwise identity between default execution, `torch.no_grad()`, and `torch.inference_mode()`.
- `test_xas_batched_vs_scalar_scale_aware`: Validates batched evaluation against scalar evaluation using physical strain-scaled tolerances.
- `test_xas_autograd_parameter_gradients`: Validates non-zero, finite autograd gradients w.r.t. $(m_1, m_2, \chi_1, \chi_2)$ against central finite differences.

---

## 3. Unresolved Blockers and Parity Status

- **Status of Full Parity**: UNRESOLVED.
- **Remaining Parity Gaps**:
  - The direct complex relative $L_2 \le 10^{-6}$ and mismatch $\le 10^{-8}$ acceptance gates remain unmet for `IMRPhenomXAS` (and downstream `IMRPhenomXHM`, `IMRPhenomXPHM`).
  - XAS continues to exhibit $3.2\% - 9.7\%$ magnitude relative $L_2$ errors against LALSimulation.
  - The finite-difference boundary input is now analytic. These tests do not establish exact continuity of the complete piecewise amplitude. Intermediate amplitude fit normalization and collocation scaling remain hypotheses for the residual, not a proven cause. Further derivation from the paper and author supplement is required; no coefficients were tuned against oracle outputs.
- **Historical Evidence**: Prior round evidence bundles remain unchanged. The editable completion report, manifest, and provenance ledger are being corrected to state the measured scope. The pre-dispatch dirty baseline was retained outside the repository.
