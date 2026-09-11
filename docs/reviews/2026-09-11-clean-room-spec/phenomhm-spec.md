# Clean-Room Mathematical Specification: PhenomHM Mapping Reconstruction

## 1. Clean-Room Boundary Rules
- Implement strictly from the mathematical formulas and literature citations documented below.
- DO NOT read, reference, import, or copy any LAL / LALSimulation / PyCBC source code, internal coefficients, or oracle fixtures.
- DO NOT fit coefficients or phase offsets to external oracle residuals.
- All code must be native PyTorch and differentiable via autograd.

## 2. Model Scope & Harvester
- Target file: `src/torchwave/approximants/imrphenom_hm.py`
- Target class: `IMRPhenomHM`
- Note: Keep `IMRPhenomHM` and `IMRPhenomXHM` distinct. `IMRPhenomXHM` should NOT just alias or wrap `IMRPhenomHM`.

## 3. Supported Modes & QNM Ringdown Fits
London et al. (2018) lists six positive-$m$ modes with $\ell \le 4$:
$$\{(\ell, m)\} = \{(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)\}$$

Fundamental QNM dimensionless frequencies and quality factors from Berti, Cardoso, Will (2006) PRD 73, 064030 Table VIII:
$$\mathcal{F}_{\ell m 0} = f_1 + f_2 (1 - j)^{f_3}, \quad Q_{\ell m 0} = q_1 + q_2 (1 - j)^{q_3}$$
where $j = a_f$ (final dimensionless spin).
$$f_{\ell m}^{\rm RD} = \frac{M}{M_f} \frac{\mathcal{F}_{\ell m 0}}{2\pi M}, \quad f_{\ell m}^{\rm damp} = \frac{f_{\ell m}^{\rm RD}}{2 Q_{\ell m 0}}$$

Parameters:
- `(2, 2)`: $f = (1.5251, -1.1568, 0.1292), \quad q = (0.7000, 1.4187, -0.4990)$
- `(2, 1)`: $f = (0.6000, -0.2339, 0.4175), \quad q = (-0.3000, 2.3561, -0.2277)$
- `(3, 3)`: $f = (1.8956, -1.3043, 0.1818), \quad q = (0.9000, 2.3430, -0.4810)$
- `(3, 2)`: $f = (1.1481, -0.5552, 0.3002), \quad q = (0.8313, 2.3773, -0.3655)$
- `(4, 4)`: $f = (2.3000, -1.5056, 0.2244), \quad q = (1.1929, 3.1191, -0.4825)$
- `(4, 3)`: $f = (1.6869, -0.8862, 0.2822), \quad q = (1.4812, 2.8096, -0.4271)$

## 4. Spin-Weighted Spherical Harmonics (${}_{-2}Y_{\ell m}$)
For positive frequencies:
$$d_+ = \frac{1}{2} \left[ {}_{-2}Y_{\ell, -m} + (-1)^\ell ({}_{-2}Y_{\ell m})^* \right]$$
$$d_\times = \frac{i}{2} \left[ {}_{-2}Y_{\ell, -m} - (-1)^\ell ({}_{-2}Y_{\ell m})^* \right]$$

For mode `(4, 3)`:
$$c_{\rm norm} = \frac{3\sqrt{14}}{16\sqrt{\pi}}$$
$${}_{-2}Y_{4, 3}(\iota, \phi_0) = c_{\rm norm} \sin\iota (1 + \cos\iota)^2 (1 - 2\cos\iota) e^{3 i \phi_0}$$
$${}_{-2}Y_{4, -3}(\iota, \phi_0) = c_{\rm norm} \sin\iota (1 - \cos\iota)^2 (1 + 2\cos\iota) e^{-3 i \phi_0}$$
$(-1)^\ell = (-1)^4 = +1$.

## 5. Piecewise Frequency Mappings (London et al. 2018)
Transition frequencies:
$$f_0^A = 0.018 \frac{f_{\ell m}^{\rm RD}}{f_{22}^{\rm RD}}, \qquad f_0^\varphi = 0.014 \frac{f_{\ell m}^{\rm RD}}{f_{22}^{\rm RD}}$$

### Amplitude frequency mapping $f_{22}^A(f)$:
$$f_{22}^A(f) = \begin{cases}
\frac{2}{m} f, & f \le f_0^A \\
\frac{f_{22}^{\rm RD} - 2 f_0^A / m}{f_{\ell m}^{\rm RD} - f_0^A} (f - f_0^A) + \frac{2 f_0^A}{m}, & f_0^A < f \le f_{\ell m}^{\rm RD} \\
f - (f_{\ell m}^{\rm RD} - f_{22}^{\rm RD}), & f > f_{\ell m}^{\rm RD}
\end{cases}$$

### Phase frequency mapping $f_{22}^\varphi(f)$:
$$f_{22}^\varphi(f) = \begin{cases}
\frac{2}{m} f, & f \le f_0^\varphi \\
\frac{f_{22}^{\rm RD} - 2 f_0^\varphi / m}{f_{\ell m}^{\rm RD} - f_0^\varphi} (f - f_0^\varphi) + \frac{2 f_0^\varphi}{m}, & f_0^\varphi < f \le f_{\ell m}^{\rm RD} \\
\frac{f_{22}^{\rm RD}}{f_{\ell m}^{\rm RD}} f, & f > f_{\ell m}^{\rm RD}
\end{cases}$$

## 6. Phase Connection & Derivative Integration
Phase derivative relationship:
$$\varphi_{\ell m}'(f) \approx \varphi_{22}'[f_{22}^\varphi(f)]$$
Integrating once:
$$\varphi_{\ell m}(f) = \kappa \varphi_{22}[f_{22}^\varphi(f)] + \Delta_{\ell m}$$
where $\kappa = \frac{1}{(f_{22}^\varphi)'(f)}$ is piecewise constant:
- **Region 1 ($f \le f_0^\varphi$):**
  $$(f_{22}^\varphi)' = \frac{2}{m} \implies \kappa = \frac{m}{2}$$
  $$\Delta_{\ell m} = \frac{\pi}{2} [3\ell + ((\ell + m) \bmod 2)] - \pi$$
- **Region 2 ($f_0^\varphi < f \le f_{\ell m}^{\rm RD}$):**
  $$s_2 = \frac{f_{22}^{\rm RD} - 2 f_0^\varphi / m}{f_{\ell m}^{\rm RD} - f_0^\varphi} \implies \kappa = \frac{1}{s_2}$$
  $$\Delta_{\ell m} = \varphi_{\ell m}(f_0^\varphi) - \kappa \varphi_{22}[f_{22}^\varphi(f_0^\varphi)]$$
  (guarantees $C^0$ continuity at $f = f_0^\varphi$)
- **Region 3 ($f > f_{\ell m}^{\rm RD}$):**
  $$s_3 = \frac{f_{22}^{\rm RD}}{f_{\ell m}^{\rm RD}} \implies \kappa = \frac{1}{s_3} = \frac{f_{\ell m}^{\rm RD}}{f_{22}^{\rm RD}}$$
  $$\Delta_{\ell m} = \varphi_{\ell m}(f_{\ell m}^{\rm RD}) - \kappa \varphi_{22}[f_{22}^\varphi(f_{\ell m}^{\rm RD})]$$
  (guarantees $C^0$ continuity at $f = f_{\ell m}^{\rm RD}$)

## 7. Amplitude Construction
The multipole amplitude is mapped from the baseline (2,2) amplitude:
$$A_{\ell m}(f) = |\beta_{\ell m}(f)| A_{22}[f_{22}^A(f)]$$
where:
$$\beta_{\ell m}(f) = \frac{\hat{H}_{\ell m}(f)}{\hat{H}_{22}[f_{22}^A(f)]}$$
with $\hat{H}_{22} \approx -1$ at leading order.

From Mishra et al. (2016) / London et al. (2018):
Let $V_m = (\pi M (2f/m))^{1/3}$ (or $V_m = (\pi M f / m)^{1/3}$), $\delta = (m_1 - m_2)/M$, $\chi_{\rm s} = (\chi_1 + \chi_2)/2$, $\chi_{\rm a} = (\chi_1 - \chi_2)/2$, $\eta = m_1 m_2 / M^2$:
- Mode `(3, 3)`:
  $$\hat{H}_{33} = -\frac{3}{4} \sqrt{\frac{5}{7}} \delta V_3$$
- Mode `(4, 4)`:
  $$\hat{H}_{44} = -\frac{4}{9} \sqrt{\frac{10}{7}} (1 - 3\eta) V_4^2$$
- Mode `(2, 1)`:
  $$\hat{H}_{21} = -\frac{\sqrt{2}}{3} \left[ \delta V_1 - \frac{3}{2} (\chi_{\rm a} + \delta \chi_{\rm s}) V_1^2 \right]$$
- Mode `(3, 2)`:
  $$\hat{H}_{32} = -\frac{1}{3} \sqrt{\frac{5}{7}} \left[ (1 - 3\eta) V_2^2 + 4\eta \chi_{\rm s} V_2^3 \right]$$
- Mode `(4, 3)`:
  $$\hat{H}_{43} = -\frac{3}{4} \sqrt{\frac{3}{35}} \left[ \delta (1 - 2\eta) V_3^3 + \frac{5}{2} \eta (\chi_{\rm a} - \delta \chi_{\rm s}) V_3^4 \right]$$

### Spin Asymmetry in Odd Modes:
Note that for $m_1 = m_2$ ($\delta = 0$), the odd modes do NOT vanish if spins are unequal ($\chi_{\rm a} \neq 0$):
For $(2,1)$, the spin term is $-\frac{3}{2} \chi_{\rm a} V_1^2 \neq 0$.
Do not force odd-$m$ modes to zero solely based on $m_1 == m_2$. Only zero when $\hat{H}_{\ell m} = 0$.

## 8. Acceptance Verification
In `tests/test_phenomhm_reconstruction.py`:
1. Piecewise mappings $f_{22}^\varphi$ and $f_{22}^A$ are strictly continuous across $f_0$ and $f_{\ell m}^{\rm RD}$.
2. Phase $\varphi_{\ell m}(f)$ is continuous across boundaries.
3. Check derivative relationship $\varphi_{\ell m}'(f) \approx \kappa \varphi_{22}'[f_{22}^\varphi(f)]$.
4. Check all 6 modes produce finite strains.
5. Check body exchange ($m_1 \leftrightarrow m_2, \chi_1 \leftrightarrow \chi_2$).
6. Check equal-mass unequal-spin produces non-zero (2,1) mode.
7. Autograd gradient checks on physical parameters.
