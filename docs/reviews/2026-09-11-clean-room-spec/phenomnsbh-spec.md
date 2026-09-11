# Work Package 5: IMRPhenomNSBH Specification

## 1. Scope & Clean-Room Boundary
This specification defines the clean-room reconstruction of the **IMRPhenomNSBH** frequency-domain gravitational waveform model for black hole–neutron star coalescences.

### Strict Clean-Room Rules
- Implement exclusively from the published literature cited below and author ancillary releases.
- **DO NOT** inspect, read, copy, or port any LAL, LALSimulation, or PyCBC source code, internal coefficients, or oracle regression fixtures.
- **DO NOT** fit coefficients or phase/time offsets to oracle residuals.
- Implement pure PyTorch native tensor math supporting autograd differentiation and batching.

### Mathematical Literature Citations
1. **Thompson et al. (2020)**: *Modeling the gravitational-wave signature of neutron-star—black-hole mergers*, [arXiv:2002.08383v2](https://arxiv.org/abs/2002.08383), Phys. Rev. D 101, 124059.
2. **Pannarale et al. (2015)**: *Phenomenological model for the gravitational-wave signal from black-hole—neutron-star coalescences*, [arXiv:1509.00512](https://arxiv.org/abs/1509.00512), Phys. Rev. D 92, 084050.
3. **Pannarale et al. (2015)**: *Cutoff frequency of the gravitational-wave signal from black-hole—neutron-star coalescences*, [arXiv:1509.06209](https://arxiv.org/abs/1509.06209), Phys. Rev. D 92, 081504.
4. **Zappa et al. (2019)**: *Black-Hole Remnants from Black-Hole—Neutron-Star Mergers*, [arXiv:1903.11622](https://arxiv.org/abs/1903.11622), Phys. Rev. Lett. 123, 041102.
5. **Jiménez-Forteza et al. (2017)**: *Hierarchical data-driven approach to fitting numerical relativity data for nonprecessing binary black holes with an application to final spin and radiated energy*, [arXiv:1611.00332](https://arxiv.org/abs/1611.00332), Phys. Rev. D 95, 064024.
6. **Santamaria et al. (2010)**: *Matching post-Newtonian and numerical relativity waveforms: systematic errors and a new phenomenological model for non-precessing black hole binaries*, [arXiv:1005.3306](https://arxiv.org/abs/1005.3306), Phys. Rev. D 82, 064016.
7. **Yagi & Yunes (2017)**: *Approximate Universal Relations for Neutron Stars and Quark Stars*, [arXiv:1608.02582](https://arxiv.org/abs/1608.02582), Phys. Rept. 681, 1-72.
8. **London et al. (2018)**: *First Higher-Harmonic Analysis of Numerical Relativity Black-Hole Mergers*, [arXiv:1708.00404](https://arxiv.org/abs/1708.00404), Phys. Rev. Lett. 120, 161102.
9. **Dietrich et al. (2019)**: *Improving the NRTidal model for binary neutron star systems*, [arXiv:1905.06011](https://arxiv.org/abs/1905.06011), Phys. Rev. D 100, 044003.

---

## 2. Parameter Definitions & Conventions
- Component masses: $m_1 = M_{\rm BH}$, $m_2 = M_{\rm NS}$ in solar masses ($M_\odot$).
- Mass ratio: $q = m_1 / m_2 \ge 1$.
- Total mass: $M = m_1 + m_2$.
- Symmetric mass ratio: $\eta = m_1 m_2 / M^2 = q / (1+q)^2$.
- Mass difference: $\delta = (m_1 - m_2) / M = \sqrt{1 - 4\eta}$.
- Aligned spin components: $\chi_{\rm BH} = \chi_{1z} \in [-0.99, 0.99]$, $\chi_{\rm NS} = \chi_{2z} \in [-0.05, 0.05]$ (typically assumed $\chi_{\rm NS} = 0$).
- Weighted spin parameter:
  $$\chi \equiv \frac{m_1}{M} \chi_1 + \frac{m_2}{M} \chi_2 = \frac{1+\delta}{2} \chi_1 + \frac{1-\delta}{2} \chi_2$$
- Tidal deformability: $\Lambda_{\rm NS} = \Lambda_2 \ge 0$. ($\Lambda_1 = 0$ for black hole).
- Geometric time scale: $M_{\rm sec} = M \times M_\odot G / c^3$, $M_{\rm BH, sec} = m_1 \times M_\odot G / c^3$.
- Distance in geometric seconds: $D_{\rm sec} = D_L \times {\rm Mpc} / c$.

---

## 3. Detailed Algorithmic Workflow

### Step 1: NS Compactness $C(\Lambda)$
From Thompson et al. (2020) Eq. 20 (Appendix B, Eq. 791):
$$
C(\Lambda) =
\begin{cases}
a_0 + a_1 \ln\Lambda + a_2 (\ln\Lambda)^2, & \Lambda > 1 \\
0.5 + (3a_0 - a_1 - 1.5)\Lambda^2 + (a_1 - 2a_0 + 1)\Lambda^3, & \Lambda \le 1
\end{cases}
$$
Coefficients:
$$a_0 = 0.360, \quad a_1 = -0.0355, \quad a_2 = 0.000705$$
Note: $C(0) = 0.5$ (Schwarzschild limit), and $C(\Lambda)$ is $C^1$-smooth across $\Lambda = 1$.

### Step 2: Relativistic Tidal Disruption Root $\xi_{\rm tide}$ and Frequency $f_{\rm tide}$
From Thompson et al. (2020) Eqs. 7–9 (Appendix A, Eqs. 641–652):
Let $\mu = q C$. The quantity $\xi_{\rm tide}$ is the largest positive real root of:
$$
P(\xi) = \xi^5 - 3\mu\xi^4 + 2\chi\sqrt{\mu^3\xi^7} - 3q\xi^2 + 6q\mu\xi - 3q\mu^2\chi^2 = 0
$$
Derivative for Newton-Raphson:
$$
P'(\xi) = 5\xi^4 - 12\mu\xi^3 + 7\chi\sqrt{\mu^3\xi^5} - 6q\xi + 6q\mu
$$
Initial guess: $\xi_0 = (3q)^{1/3}$.
Iterate $\xi_{n+1} = \xi_n - P(\xi_n)/P'(\xi_n)$ for 10–15 steps (or until $|P(\xi)| < 10^{-12}$).
Then:
$$
\tilde{r}_{\rm tide} = \xi_{\rm tide} M_{\rm BH, sec} \frac{1 - 2C}{\mu} = \xi_{\rm tide} M_{\rm NS, sec} \frac{1 - 2C}{C}
$$
Tidal disruption frequency in Hz:
$$
f_{\rm tide} = \frac{1}{\pi \left( \chi M_{\rm BH, sec} + \sqrt{\tilde{r}_{\rm tide}^3 / M_{\rm BH, sec}} \right)}
$$

### Step 3: Baryonic Torus Remnant Mass Ratio $M_{\rm b, torus}/M_{\rm b, NS}$
From Foucart et al. (2012) / Thompson et al. (2020) Eqs. 10 & 656–669:
Kerr ISCO radius of unit mass BH:
$$
Z_1 = 1 + (1 - \chi^2)^{1/3} \left[ (1 + \chi)^{1/3} + (1 - \chi)^{1/3} \right]
$$
$$
Z_2 = \sqrt{3\chi^2 + Z_1^2}
$$
$$
\bar{r}_{\rm ISCO} = 3 + Z_2 - {\rm sign}(\chi)\sqrt{(3 - Z_1)(3 + Z_1 + 2 Z_2)}
$$
(with ${\rm sign}(\chi) = 1$ if $\chi \ge 0$ else $-1$).
Torus remnant mass ratio:
$$
\frac{M_{\rm b, torus}}{M_{\rm b, NS}} = \max\left(0.0, \; 0.296 \xi_{\rm tide}(1 - 2C) - 0.171 q C \bar{r}_{\rm ISCO}\right)
$$

### Step 4: Remnant Mass $M_f$ and Remnant Spin $\chi_f$
From Zappa et al. (2019) Eq. 6 & Table IV:
$$
F(\eta, \chi, \Lambda) = F_{\rm BBH}(\eta, \chi) \frac{1 + p_1(\eta, \chi)\Lambda + p_2(\eta, \chi)\Lambda^2}{\left(1 + [p_3(\eta, \chi)]^2 \Lambda\right)^2}
$$
where
$$
p_k(\eta, \chi) = (p_{k10}\chi + p_{k11})\eta + (p_{k20}\chi + p_{k21})\eta^2
$$
$F_{\rm BBH}(\eta, \chi)$ is the Jiménez-Forteza et al. (2017) v2 model for $M_f/M$ and $\chi_f$ (`FinalStateUIB2016`):
- For $M_f/M$ ($X_\bullet$):
  - $k=1$: $p_{110} = -1.83 \times 10^{-3}$, $p_{111} = 2.39 \times 10^{-3}$, $p_{120} = 4.29 \times 10^{-3}$, $p_{121} = 9.80 \times 10^{-3}$
  - $k=2$: $p_{210} = 2.34 \times 10^{-7}$, $p_{211} = -8.28 \times 10^{-7}$, $p_{220} = -1.64 \times 10^{-6}$, $p_{221} = 8.08 \times 10^{-6}$
  - $k=3$: $p_{310} = -2.01 \times 10^{-2}$, $p_{311} = 1.32 \times 10^{-1}$, $p_{320} = 6.51 \times 10^{-2}$, $p_{321} = -1.43 \times 10^{-1}$
- For $\chi_f$ ($a_\bullet$):
  - $k=1$: $p_{110} = -5.44 \times 10^{-3}$, $p_{111} = 7.91 \times 10^{-3}$, $p_{120} = 2.33 \times 10^{-2}$, $p_{121} = 2.48 \times 10^{-2}$
  - $k=2$: $p_{210} = -8.57 \times 10^{-7}$, $p_{211} = -2.82 \times 10^{-6}$, $p_{220} = 6.61 \times 10^{-6}$, $p_{221} = 4.29 \times 10^{-5}$
  - $k=3$: $p_{310} = -3.04 \times 10^{-2}$, $p_{311} = 2.55 \times 10^{-1}$, $p_{320} = 1.48 \times 10^{-1}$, $p_{321} = -4.28 \times 10^{-1}$
Physical bounds: $M_f = \min(M, M \cdot (M_f/M))$, $\chi_f = {\rm clamp}(\chi_f, -0.999, 0.999)$.

### Step 5: Kerr Ringdown Frequency $f_{\rm RD}$ and Quality Factor $Q$
From London et al. (2018) / Thompson et al. (2020) Eqs. 14–17 (Eqs. 692–705):
$$
\kappa(\chi_f) = \sqrt{\frac{\ln(2 - \chi_f)}{\ln 3}}
$$
$$
\tilde{\omega}(\kappa) = 1.0 + 1.5578 e^{2.9031 i} \kappa + 1.9510 e^{5.9210 i} \kappa^2 + 2.0997 e^{2.7606 i} \kappa^3 + 1.4109 e^{5.9143 i} \kappa^4 + 0.4106 e^{2.7952 i} \kappa^5
$$
In Hz:
$$
f_{\rm RD} = \frac{\Re(\tilde{\omega})}{2 \pi M_{f, \rm sec}}
$$
$$
Q = \frac{\Re(\tilde{\omega})}{2 \Im(\tilde{\omega})}
$$

### Step 6: Merger Classification and Joining Parameters
Scaled ringdown frequency:
$$
\tilde{f}_{\rm RD} =
\begin{cases}
0.99 \times 0.98 f_{\rm RD}, & \Lambda > 1 \\
(1 - 0.02\Lambda + 0.01\Lambda^2) \times 0.98 f_{\rm RD}, & \Lambda \le 1
\end{cases}
$$
Auxiliary variables (Eqs. 717–728):
$$
x_{\rm ND} = \left(\frac{f_{\rm tide} - \tilde{f}_{\rm RD}}{\tilde{f}_{\rm RD}}\right)^2 - 0.571505 C - 0.00508451 \chi
$$
$$
x'_{\rm ND} = \left(\frac{f_{\rm tide} - \tilde{f}_{\rm RD}}{\tilde{f}_{\rm RD}}\right)^2 - 0.657424 C - 0.0259977 \chi
$$
$$
x_{\rm D} = \frac{M_{\rm b, torus}}{M_{\rm b, NS}} + 0.424912 C + 0.363604 \sqrt{\eta} - 0.060559 \chi
$$
$$
x'_{\rm D} = \frac{M_{\rm b, torus}}{M_{\rm b, NS}} - 0.132754 C + 0.576669 \sqrt{\eta} - 0.0603749 \chi - 0.0601185 \chi^2 - 0.0729134 \chi^3
$$

Hyperbolic tangent window:
$$
\omega^\pm_{f_0, d}(f) = \frac{1}{2}\left[ 1 \pm \tanh\left(\frac{4(f - f_0)}{d}\right) \right]
$$

Table 1 Quantities:
| Quantity | Non-disruptive ($f_{\rm tide} \ge f_{\rm RD}$, $M_{\rm b,torus} = 0$) | Mildly disruptive ($f_{\rm tide} \ge f_{\rm RD}$, $M_{\rm b,torus} > 0$) | Mildly disruptive ($f_{\rm tide} < f_{\rm RD}$, $M_{\rm b,torus} = 0$) | Disruptive ($f_{\rm tide} < f_{\rm RD}$, $M_{\rm b,torus} > 0$) |
|---|---|---|---|---|
| $\epsilon_{\rm tide}$ | $\omega^+_{-0.0796251, 0.0801192}(x_{\rm ND})$ | $\omega^+_{-0.0796251, 0.0801192}(x_{\rm ND})$ | $0.0$ | $0.0$ |
| $\epsilon_{\rm ins}$ | $1.0$ | $1.29971 - 1.61724 x_{\rm D}$ | $1.29971 - 1.61724 x_{\rm D}$ | $1.29971 - 1.61724 x_{\rm D}$ |
| $\sigma_{\rm tide}$ | $\omega^-_{-0.206465, 0.226844}(x'_{\rm ND})$ | $\omega^-_{-0.206465, 0.226844}(x'_{\rm ND})$ | $(\omega^-_{x_2, d_2}(x'_{\rm ND}) + 0.137722 - 0.293237 x'_{\rm D})/2$ | $0.137722 - 0.293237 x'_{\rm D}$ |
| $\tilde{f}_0$ | $\tilde{f}_{\rm RD}$ | $\epsilon_{\rm ins}\tilde{f}_{\rm RD}$ | $[(q-1)\tilde{f}_{\rm RD} + \epsilon_{\rm ins}f_{\rm tide}]/q$ | $\epsilon_{\rm ins} f_{\rm tide}$ |
| $\tilde{f}_1$ | $\tilde{f}_{\rm RD}$ | $\epsilon_{\rm ins}\tilde{f}_{\rm RD}$ | $[(q-1)\tilde{f}_{\rm RD} + f_{\rm tide}]/q$ | $f_{\rm tide}$ |
| $\tilde{f}_2$ | $\tilde{f}_{\rm RD}$ | $\tilde{f}_{\rm RD}$ | $-$ | $-$ |

Window width: $d_{\rm win} = (0.015 + \sigma_{\rm tide}) f_{\rm RD}$.

### Step 7: Phenomenological Amplitude Assembly
From Santamaria et al. (2010) Table II:
For $\Lambda_k \in \{\gamma_1, \delta_1, \delta_2\}$:
$$
\Lambda_k = \zeta^{(01)}\chi + \zeta^{(02)}\chi^2 + \zeta^{(11)}\eta\chi + \zeta^{(10)}\eta + \zeta^{(20)}\eta^2
$$
- $\gamma_1$: $\zeta = (4.149, -4.07, -87.52, -48.97, 666.5)$
- $\delta_1$: $\zeta = (-0.05472, 0.02094, 0.3554, 0.1151, 0.964)$
- $\delta_2$: $\zeta = (-1.235, 0.3423, 6.062, 5.949, -10.69)$
Constraints: $\gamma_1 = \max(0.0, \gamma_1)$, $\delta_1 = \max(0.0, \delta_1)$, $\delta_2 = \max(10^{-4}, \delta_2)$.

Correction parameters (Thompson et al. Eqs. 743 & 760):
$$
\gamma'_1 =
\begin{cases}
1.25, & \Lambda > 1 \\
1 - 0.5\Lambda - 0.25\Lambda^2, & \Lambda \le 1
\end{cases}
\times \gamma_1
$$
$$
\delta'_2 =
\begin{cases}
\frac{1.62496}{2}\omega^-_{0.0188092, 0.338737}\left(\frac{f_{\rm tide} - \tilde{f}_{\rm RD}}{\tilde{f}_{\rm RD}}\right), & \Lambda > 1 \\
\delta_2 - 2(\delta_2 - 0.81248)\Lambda + (\delta_2 - 0.81248)\Lambda^2, & \Lambda \le 1
\end{cases}
$$
Lorentzian damping width: $\sigma = \delta'_2 f_{\rm RD} / Q$.

Component Amplitudes:
1. **Inspiral $A_{\rm PN}(f)$**:
   Dimensionless frequency $v = (\pi M_{\rm sec} f)^{1/3}$, $x = v^2$.
   TaylorT4 expansion $\dot{x} = \frac{64}{5}\eta x^5 \sum_{k=0}^7 a_k x^{k/2}$ with $a_k$ from Santamaria et al. Eq. 2143–2174.
   Blanchet 3PN amplitude $\sum_{k=0}^6 \mathcal{A}_k x^{k/2}$ (real parts) from Santamaria et al. Eq. 2235–2248:
   - $a_0 = 1$, $a_1 = 0$, $a_2 = -743/336 - 11\eta/4$, $a_3 = 4\pi - 113\chi/12 + 19\eta(\chi_1+\chi_2)/6$
   - $a_4 = 34103/18144 + 5\chi^2 + \eta(13661/2016 - \chi_1\chi_2/8) + 59\eta^2/18$
   - $a_5 = -\pi(4159/672 + 189\eta/8) - \chi(31571/1008 - 1165\eta/24) + (\chi_1+\chi_2)(21863\eta/1008 - 79\eta^2/6) - 3\chi^3/4 + 9\eta\chi\chi_1\chi_2/4$
   - $a_6 = 16447322263/139708800 - 1712\gamma_E/105 + 16\pi^2/3 - (856/105)\ln(16x) + \eta(451\pi^2/48 - 56198689/217728) + 541\eta^2/896 - 5605\eta^3/2592 - 80\pi\chi/3 + (20\pi/3 - 1135\chi/36)\eta(\chi_1+\chi_2) + (64153/1008 - 457\eta/36)\chi^2 - (787\eta/144 - 3037\eta^2/144)\chi_1\chi_2$
   - $a_7 = -\pi(4415/4032 - 358675\eta/6048 - 91495\eta^2/1512) - \chi(2529407/27216 - 845827\eta/6048 + 41551\eta^2/864) + (\chi_1+\chi_2)(1580239\eta/54432 - 451597\eta^2/6048 + 2045\eta^3/432 + 107\eta\chi^2/6 - 5\eta^2\chi_1\chi_2/24) + 12\pi\chi^2 - \chi^3(1505/24 + \eta/8) + \chi\chi_1\chi_2(101\eta/24 + 3\eta^2/8)$
   - $\mathcal{A}_0 = 1$, $\mathcal{A}_1 = 0$, $\mathcal{A}_2 = -107/42 + 55\eta/42$, $\mathcal{A}_3 = 2\pi - 4\chi/3 + 2\eta(\chi_1+\chi_2)/3$
   - $\mathcal{A}_4 = -2173/1512 - \eta(1069/216 - 2\chi_1\chi_2) + 2047\eta^2/1512$
   - $\mathcal{A}_5 = -107\pi/21 + 34\pi\eta/21$
   - $\mathcal{A}_6 = 27027409/646800 - 856\gamma_E/105 + 2\pi^2/3 + \eta(41\pi^2/96 - 278185/33264) - 20261\eta^2/2772 + 114635\eta^3/99792 - (428/105)\ln(16x)$
   $$
   A_{\rm PN}(f) = \frac{M_{\rm sec}^2}{D_{\rm sec}} \pi \sqrt{\frac{2\eta}{3}} (\pi M_{\rm sec} f)^{-7/6} \frac{\sum_{k=0}^6 \mathcal{A}_k x^{k/2}}{\sqrt{\sum_{k=0}^7 a_k x^{k/2}}}
   $$
2. **Pre-merger $A_{\rm PM}(f)$**:
   $$
   A_{\rm PM}(f) = \frac{M_{\rm sec}^2}{D_{\rm sec}} \gamma'_1 (\pi M_{\rm sec} f)^{5/6} (\pi M_{\rm sec})^{-7/6}
   $$
3. **Ringdown $A_{\rm RD}(f)$**:
   $$
   A_{\rm RD}(f) = \frac{M_{\rm sec}^2}{D_{\rm sec}} \epsilon_{\rm tide} \delta_1 \frac{\sigma^2}{(f - f_{\rm RD})^2 + \sigma^2/4} (\pi M_{\rm sec} f)^{-7/6}
   $$

Total Amplitude:
$$
A(f) = A_{\rm PN}(f) \omega^-_{\tilde{f}_0, d_{\rm win}}(f) + A_{\rm PM}(f) \omega^-_{\tilde{f}_1, d_{\rm win}}(f) + A_{\rm RD}(f) \omega^+_{\tilde{f}_2, d_{\rm win}}(f)
$$

### Step 8: Phase Model & Waveform Synthesis
From Thompson et al. (2020) Sec. IIC:
$$
\phi(f) = \phi_{\rm PhenomD}(f) + \phi_{\rm NRTidalv2}(f)
$$
where $\phi_{\rm PhenomD}$ is the aligned-spin BBH phase from `IMRPhenomD`, and $\phi_{\rm NRTidalv2}$ is the tidal phase from `IMRPhenomD_NRTidalv2` (anchored at reference frequency $f_{\rm ref}$).
The 22-mode complex strain is:
$$
h_{22}(f) = A(f) e^{-i \phi(f)}
$$
Polarizations $h_+(f)$ and $h_\times(f)$ are formed by projecting onto spin-weighted spherical harmonics ${}_{-2}Y_{2, \pm 2}(\iota, \pi/2 - \varphi_{\rm ref})$.

---

## 4. Acceptance Criteria
1. **Compactness & Disruption Roots**:
   - $C(0) = 0.5$, $C(\Lambda)$ strictly monotonically decreasing for $\Lambda > 0$.
   - Newton solver for $\xi_{\rm tide}$ converges to residual $|P(\xi)| < 10^{-12}$.
   - For high mass ratio non-disrupting systems ($q=6, \Lambda=500$), verify $f_{\rm tide} > f_{\rm RD}$ and classified as non-disruptive.
   - For low mass ratio disrupting systems ($q=1.5, \Lambda=800, \chi=0.5$), verify $f_{\rm tide} < f_{\rm RD}$ and classified as disruptive with $M_{\rm b,torus} > 0$.
2. **Remnant Limits**:
   - Remnant mass $M_f \le M$ and remnant spin $|\chi_f| < 1.0$.
   - At $\Lambda \to 0$, $M_f$ and $\chi_f$ reproduce the BBH limit $F_{\rm BBH}$.
3. **Amplitude Joining & Smoothness**:
   - Verify $A(f) > 0$ across all frequency bins for physical parameters ($q \in [1, 15]$, $|\chi| \le 0.8$, $\Lambda \in [0, 5000]$).
   - In the disruptive regime, verify high-frequency exponential suppression beyond $f_{\rm tide}$.
4. **Consistency & Parity**:
   - Batched and scalar evaluations agree to within machine precision ($< 10^{-14}$).
   - Autograd gradients with respect to $m_1, m_2, \chi_1, \Lambda_2$, and $\iota$ are finite, non-zero, and free of NaNs.
   - All tests in `tests/test_phenomnsbh_reconstruction.py` pass.
   - Full test suite (320+ tests) remains green.
