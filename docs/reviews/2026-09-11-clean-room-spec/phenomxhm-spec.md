# Clean-Room Mathematical Specification: IMRPhenomXHM Calibrated Higher Modes

## 1. Provenance and Scope

This specification defines the clean-room, paper-based reconstruction of the calibrated higher-order spherical harmonic modes for the frequency-domain gravitational waveform model `IMRPhenomXHM`.

### Primary Literature References
- **García-Quirós et al. (2020)**: *"Multimode frequency-domain model for the gravitational wave signal from non-precessing black-hole binaries: IMRPhenomXHM"*, Phys. Rev. D 102, 064002, [arXiv:2001.10914v2](https://arxiv.org/abs/2001.10914).
- **Pratten et al. (2020)**: *"Setting the cornerstone for next-generation models of gravitational reactions: IMRPhenomXAS"*, Phys. Rev. D 101, 024029, [arXiv:2001.10911](https://arxiv.org/abs/2001.10911).
- **Author Ancillary Release**: Mathematical and coefficient data release at [arXiv:2001.10914v2/anc](https://arxiv.org/src/2001.10914v2/anc). All 88 Wolfram `.m` fit files are recorded in [`docs/reviews/2026-09-11-clean-room-spec/xhm-ancillary-sha256.json`](file:///Users/xangma/repos/torchwave/docs/reviews/2026-09-11-clean-room-spec/xhm-ancillary-sha256.json) and evaluated through the pure-PyTorch fit library in [`src/torchwave/approximants/xhm_fits.py`](file:///Users/xangma/repos/torchwave/src/torchwave/approximants/xhm_fits.py).

### Clean-Room Rules
- All code must be fresh PyTorch implementations derived directly from this specification and `XHMParameterFits`.
- Zero reading, copying, or porting from LAL/LALSimulation/PyCBC source code or internal constants.
- No empirical tuning of coefficients to oracle residuals.

---

## 2. Intrinsic Binary and Spin Variables

Let component masses be $m_1 \ge m_2$, total mass $M = m_1 + m_2 = 1$ in geometric units ($G = c = M = 1$), symmetric mass ratio $\eta = m_1 m_2 / M^2$, and mass difference parameter $\delta = \sqrt{1 - 4\eta} = (m_1 - m_2)/M$.
Dimensionless aligned spins are $\chi_1, \chi_2 \in [-1, 1]$.
Spin combinations:
$$\chi_{\rm s} = \frac{1}{2}(\chi_1 + \chi_2), \quad \chi_{\rm a} = \frac{1}{2}(\chi_1 - \chi_2)$$
$$\chi_{\rm eff} = m_1 \chi_1 + m_2 \chi_2$$
The mass-weighted effective total spin $S$ is:
$$S = \frac{m_1^2 \chi_1 + m_2^2 \chi_2}{1 - 2\eta}$$
The remnant final spin $\chi_{\rm f}$ and final mass $M_{\rm f}$ are evaluated using `XASParameterFits.fit_final_spin(eta, chi1, chi2)` and `XASParameterFits.fit_erad(eta, chi1, chi2)` ($M_{\rm f} = 1 - E_{\rm rad}$).

---

## 3. Supported Modes and Conventions

Supported positive-$m$ modes:
$$\{(\ell, m)\} = \{(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)\}$$
The carrier mode $(2, 2)$ is computed directly using `IMRPhenomXAS`.
Negative-$m$ modes follow equatorial reflection symmetry:
$$h_{\ell, -m}(f) = (-1)^\ell h_{\ell m}^*(f)$$
The full complex polarizations are:
$$h_+(f) - i h_\times(f) = \sum_{\ell, m} h_{\ell m}(f) {}_{-2}Y_{\ell m}(\iota, \beta)$$
where $\beta = \frac{\pi}{2} - \varphi_{\rm ref}$ and $\iota$ is inclination.

---

## 4. Frequency Regimes and Boundaries

Each mode $(\ell, m)$ is partitioned into three frequency domains:
1. **Inspiral**: $f \in [f_{\rm min}, f_{\rm Ins}^{\ell m}]$
2. **Intermediate**: $f \in [f_{\rm Ins}^{\ell m}, f_{\rm RD}^{\ell m}]$
3. **Ringdown**: $f \in [f_{\rm RD}^{\ell m}, f_{\rm max}]$

### Boundary Frequencies

#### Inspiral Cutting Frequency $f_{\rm Ins}^{\ell m}$
- **For Phase**:
  $$f_{\rm Ins, \phi}^{\ell m} = \frac{m}{2} f_{\rm MECO}^{22}$$
- **For Amplitude**:
  $$f_{\rm Ins, A}^{\ell m} = \frac{m}{2} \left( f_{\rm MECO}^{22} \epsilon_{\rm Ins}^{\ell m} + |f_{\rm ISCO}^{22} - f_{\rm MECO}^{22}| \delta_{\rm Ins}^{\ell m} \right)$$
  where:
  - $(2, 1)$: $\epsilon = 1.0$, $\delta = \frac{3}{4} - 0.235 \chi_{\rm eff} - \frac{5}{6} \chi_{\rm eff}^2$
  - $(3, 3)$: $\epsilon = 1.0$, $\delta = \frac{3}{4} - 0.235 \chi_{\rm eff} - \frac{5}{6} \chi_{\rm eff}$
  - $(3, 2)$: $\epsilon = f_{\rm ring}^{32} / f_{\rm ring}^{22}$, $\delta = \left(\frac{3}{4} - 0.235 |\chi_{\rm eff}|\right) \frac{f_{\rm ring}^{32}}{f_{\rm ring}^{22}}$
  - $(4, 4)$: $\epsilon = 1.0$, $\delta = \frac{3}{4} - 0.235 \chi_{\rm eff}$

#### Ringdown Cutting Frequency $f_{\rm RD}^{\ell m}$
- **For Amplitude**:
  - $(2, 1)$: $f_{\rm RD, A}^{21} = 0.75 f_{\rm ring}^{21}$
  - $(3, 3)$: $f_{\rm RD, A}^{33} = 0.95 f_{\rm ring}^{33}$
  - $(3, 2)$: $f_{\rm RD, A}^{32} = f_{\rm ring}^{22}$
  - $(4, 4)$: $f_{\rm RD, A}^{44} = 0.9 f_{\rm ring}^{44}$
- **For Phase**:
  - $(2, 1)$: $f_{\rm RD, \phi}^{21} = f_{\rm ring}^{21} - f_{\rm damp}^{21}$
  - $(3, 3)$: $f_{\rm RD, \phi}^{33} = f_{\rm ring}^{33} - f_{\rm damp}^{33}$
  - $(3, 2)$: $f_{\rm RD, \phi}^{32} = f_{\rm ring}^{22} - 0.5 f_{\rm damp}^{22}$
  - $(4, 4)$: $f_{\rm RD, \phi}^{44} = f_{\rm ring}^{44} - f_{\rm damp}^{44}$

---

## 5. Amplitude Calibration

For each mode $(\ell, m)$, the amplitude is normalized by the leading-order quadrupole amplitude:
$$A_{22}^0(f) = \pi \sqrt{\frac{2\eta}{3}} (\pi f)^{-7/6}$$
We model the dimensionless ratio $\mathcal{H}_{\ell m}(f) = |A_{\ell m}(f)| / A_{22}^0(f)$.

### 5.1 Inspiral Amplitude ($f \le f_{\rm Ins}^{\ell m}$)
$$\mathcal{H}_{\ell m}^{\rm Ins}(f) = \frac{|A_{\ell m}^{\rm PN}(f)|}{A_{22}^0(f)} + \alpha \left(\frac{f}{f_{\rm Ins}}\right)^{7/3} + \beta \left(\frac{f}{f_{\rm Ins}}\right)^{8/3} + \gamma \left(\frac{f}{f_{\rm Ins}}\right)^3$$
where:
$$\frac{|A_{\ell m}^{\rm PN}(f)|}{A_{22}^0(f)} = \left(\frac{2}{m}\right)^{7/6} |\hat{H}_{\ell m}(v)|, \quad v = \left(\frac{2\pi f}{m}\right)^{1/3}$$
The expansion functions $\hat{H}_{\ell m}$ are given by the 3PN formulas from García-Quirós et al. (2020) Appendix E (Eqs. 1864-1960).
Crucially, for odd-$m$ modes ((2,1), (3,3)), $\hat{H}_{\ell m}$ contains terms proportional to $\delta$ and terms proportional to $\chi_a = (\chi_1 - \chi_2)/2$. Thus, when $\delta = 0$ (equal masses $m_1 = m_2$), the odd modes **do not vanish** if $\chi_1 \neq \chi_2$!

The coefficients $(\alpha, \beta, \gamma)$ are determined by matching $\mathcal{H}_{\ell m}(f)$ to the 3 collocation points:
$f_1 = 0.5 f_{\rm Ins}, f_2 = 0.75 f_{\rm Ins}, f_3 = f_{\rm Ins}$,
where $\mathcal{H}_i^{\rm fit}$ are given by `XHMParameterFits.amp_insp1`, `amp_insp2`, `amp_insp3`.
$$\begin{pmatrix} 0.5^{7/3} & 0.5^{8/3} & 0.5^3 \\ 0.75^{7/3} & 0.75^{8/3} & 0.75^3 \\ 1 & 1 & 1 \end{pmatrix} \begin{pmatrix} \alpha \\ \beta \\ \gamma \end{pmatrix} = \begin{pmatrix} \mathcal{H}_1^{\rm fit} - \mathcal{H}_1^{\rm PN} \\ \mathcal{H}_2^{\rm fit} - \mathcal{H}_2^{\rm PN} \\ \mathcal{H}_3^{\rm fit} - \mathcal{H}_3^{\rm PN} \end{pmatrix}$$

### 5.2 Intermediate Amplitude ($f_{\rm Ins}^{\ell m} \le f \le f_{\rm RD}^{\ell m}$)
$$\frac{A_0}{A_{\ell m}^{\rm Inter}(f)} = \sum_{k=0}^5 \delta_k f^k, \quad A_0 = \pi \sqrt{\frac{2\eta}{3}}$$
The 6 coefficients $\mathbf{\delta} = (\delta_0, \dots, \delta_5)^T$ are solved by matching:
1. $P(f_{\rm Ins}) = A_0 / A_{\rm Ins}(f_{\rm Ins})$
2. $P'(f_{\rm Ins}) = - A_0 A_{\rm Ins}'(f_{\rm Ins}) / [A_{\rm Ins}(f_{\rm Ins})]^2$
3. $P(f_{\rm Int1}) = A_0 / [A_{22}^0(f_{\rm Int1}) \mathcal{H}_{\rm Int1}^{\rm fit}]$ where $f_{\rm Int1} = f_{\rm Ins} + \frac{1}{3}(f_{\rm RD} - f_{\rm Ins})$
4. $P(f_{\rm Int2}) = A_0 / [A_{22}^0(f_{\rm Int2}) \mathcal{H}_{\rm Int2}^{\rm fit}]$ where $f_{\rm Int2} = f_{\rm Ins} + \frac{2}{3}(f_{\rm RD} - f_{\rm Ins})$
5. $P(f_{\rm RD}) = A_0 / A_{\rm RD}(f_{\rm RD})$
6. $P'(f_{\rm RD}) = - A_0 A_{\rm RD}'(f_{\rm RD}) / [A_{\rm RD}(f_{\rm RD})]^2$
This guarantees exact $C^1$ continuity at both patch joins.

### 5.3 Ringdown Amplitude ($f \ge f_{\rm RD}^{\ell m}$)
$$\frac{A_{\rm RD}^{\ell m}(f)}{A_{22}^0(f)} = \frac{1}{f^{1/12}} \frac{|a_\lambda| f_{\rm damp}^{\ell m} \sigma}{(f - f_{\rm ring}^{\ell m})^2 + (f_{\rm damp}^{\ell m} \sigma)^2} \exp\left( - \frac{(f - f_{\rm ring}^{\ell m}) \lambda}{f_{\rm damp}^{\ell m} \sigma} \right)$$
where $a_\lambda, \lambda$ come from `amp_alambda` and `amp_lambda`, and:
- $(3, 3)$: $\sigma = 1.3$
- $(3, 2)$: $\sigma = 1.33$ (fitted in spheroidal basis)
- $(4, 4)$: $\sigma = 1.33$
- $(2, 1)$: $\sigma$ is evaluated from `amp_sigma_21`

---

## 6. Phase Calibration

### 6.1 Inspiral Phase ($f \le f_{\rm Ins, \phi}^{\ell m}$)
$$\phi_{\ell m}^{\rm Ins}(f) = \frac{m}{2} \phi_{22}^X\left(\frac{2}{m} f\right) + \Lambda_{\ell m}(f) + d\phi_{\rm Ins} f + \phi_{\rm Ins}$$
where:
- $\phi_{22}^X$ is the XAS carrier phase.
- $\Lambda_{\ell m}(f) = \text{phase\_lambda}_{\ell m}$ (for $q \ge 100$ or default).
- $d\phi_{\rm Ins}$ and $\phi_{\rm Ins}$ are $C^1$ alignment constants matching the Intermediate Phase at $f_{\rm Ins, \phi}^{\ell m}$.

### 6.2 Intermediate Phase ($f_{\rm Ins, \phi}^{\ell m} \le f \le f_{\rm RD, \phi}^{\ell m}$)
For modes without mode-mixing ((2,1), (3,3), (4,4)), $a_3 = 0$:
$$\frac{d\phi_{\ell m}^{\rm Int}}{df} = a_\lambda \frac{f_{\rm damp}^{\ell m}}{(f_{\rm damp}^{\ell m})^2 + (f - f_{\rm ring}^{\ell m})^2} + a_0 + \frac{a_1}{f} + \frac{a_2}{f^2} + \frac{a_4}{f^4}$$
Integrating analytically gives:
$$\phi_{\ell m}^{\rm Int}(f) = a_\lambda \arctan\left(\frac{f - f_{\rm ring}^{\ell m}}{f_{\rm damp}^{\ell m}}\right) + a_0 f + a_1 \log f - \frac{a_2}{f} - \frac{a_4}{3 f^3} + C_{\rm int}$$
The 5 coefficients $(a_\lambda, a_0, a_1, a_2, a_4)$ are solved by matching $\frac{d\phi}{df}$ at 5 collocation points $f_i$ (Table VII of paper) to values $\mathcal{F}_i$ from `phase_p1` to `phase_p5`.

For mode (3,2), $a_3 \neq 0$:
$$\frac{d\phi_{32}^{\rm Int}}{df} = a_\lambda \frac{f_{\rm damp}^{32}}{(f_{\rm damp}^{32})^2 + (f - f_{\rm ring}^{32})^2} + \sum_{k=0}^4 \frac{a_k}{f^k}$$
whose 6 coefficients are fixed by matching 4 intermediate collocation points plus $C^1$ and $C^2$ continuity with the ringdown phase at $f_{\rm RD}^{32}$.

### 6.3 Ringdown Phase for Modes Without Mixing ((2,1), (3,3), (4,4))
$$\phi_{\ell m}^{\rm RD}(f) = -\alpha_2^{\ell m} \frac{(f_{\rm ring}^{\ell m})^2}{f} + \alpha_\lambda^{\ell m} \arctan\left(\frac{f - f_{\rm ring}^{\ell m}}{f_{\rm damp}^{\ell m}}\right) + d\phi_{\rm RD}^{\ell m} f + \phi_{\rm RD}^{\ell m}$$
where:
$$\alpha_\lambda^{\ell m} = \alpha_\lambda^{22} = \text{phase\_rd\_alpha\_lambda}(\eta, \chi_1, \chi_2, S)$$
$$\alpha_2^{\ell m} = \frac{f_{\rm damp}^{22}}{f_{\rm damp}^{\ell m}} \alpha_2^{22} = \frac{f_{\rm damp}^{22}}{f_{\rm damp}^{\ell m}} \text{phase\_rd\_alpha2}(\eta, \chi_1, \chi_2, S)$$
The shifts $d\phi_{\rm RD}^{\ell m}$ and $\phi_{\rm RD}^{\ell m}$ guarantee $C^1$ continuity with the intermediate phase at $f_{\rm RD, \phi}^{\ell m}$.

---

## 7. Mode (3,2) Spherical-Spheroidal Mixing

Due to frame dragging and Kerr QNM mixing between $\ell=2$ and $\ell=3$ for $m=2$:
1. The ringdown phase derivative is constructed in the spheroidal basis:
   $$\frac{d\phi_{32,S}}{df} = \frac{\alpha_2^{32}}{f^2} + \frac{\alpha_4^{32}}{f^4} + \alpha_\lambda^{32} \frac{f_{\rm damp}^{32}}{(f_{\rm damp}^{32})^2 + (f - f_{\rm ring}^{32})^2} + d\phi_{\rm RD}^{32}$$
   $$\phi_{32,S}(f) = -\frac{\alpha_2^{32}}{f} - \frac{\alpha_4^{32}}{3 f^3} + \alpha_\lambda^{32} \arctan\left(\frac{f - f_{\rm ring}^{32}}{f_{\rm damp}^{32}}\right) + d\phi_{\rm RD}^{32} f + \phi_{\rm RD}^{32}$$
   where $\alpha_2^{32} = \text{phase\_rd\_alpha2}(\eta, \chi_1, \chi_2, S)$.
2. Alignment to (2,2) carrier:
   $$\phi_{32,S}'(f_0) = \phi_{22}'(f_0) + \Delta T_{32,S}, \quad f_0 = f_{\rm ring}^{22} + f_{\rm damp}^{22}$$
   $$\phi_{32,S}(f_1) = \phi_{22}(f_1) + \Delta \phi_{32,S}, \quad f_1 = f_{\rm ring}^{22}$$
   with $\Delta T_{32,S} = \text{phase\_32\_s\_deltadphi}$ and $\Delta \phi_{32,S} = \text{phase\_32\_s\_deltaphi}$.
3. Spheroidal strain:
   $$h_{32}^S(f) = A_{32}^S(f) e^{-i \phi_{32,S}(f)}$$
   where $A_{32}^S(f)$ is the ringdown amplitude with $\sigma = 1.33$.
4. Transformation to spherical harmonic basis:
   For $f > f_{\rm RD, A}^{32}$:
   $$h_{32}(f) = \alpha_{322}(\chi_{\rm f}) h_{22}^S(f) + \alpha_{332}(\chi_{\rm f}) h_{32}^S(f)$$
   where $h_{22}^S(f) \approx h_{22}(f)$ is the dominant (2,2) mode strain, and $\alpha_{322}, \alpha_{332}$ are the complex mixing coefficients from `mixing_alpha322` and `mixing_alpha333`.
   For $f \le f_{\rm RD, A}^{32}$:
   $$h_{32}(f) = A_{32}^{\rm IMR}(f) e^{-i \phi_{32}^{\rm IMR}(f)}$$
