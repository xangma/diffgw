# Mathematical Calibration & Parity Specification for Gravitational Waveform Approximants

**Document ID**: `SPEC-TW-2026-09-11-PARITY`  
**Status**: Clean-Room Mathematical Reference  
**License**: MIT License  
**Scope**: Exact IEEE FP64 constants, phase anchoring, mode alignment, and precession reference-frame geometry for TorchWave.

---

## Clean-Room Provenance Statement

This document contains raw mathematical constants, analytical derivations, and alignment equations extracted from primary physical theory and author reference files. In accordance with strict clean-room protocol:
- No C source code, control structures, macros, typedefs, or software design structures from LALSuite are transplanted.
- All numbers are presented as IEEE-754 double-precision floating-point values (FP64) or irreducible rational fractions.
- All relationships are formulated as exact mathematical functions operating on physical or dimensionless quantities.
- The output maintains pure MIT license integrity.

---

## 1. NRTidal and NRTidalv2 Models

### 1.1 Physical Coordinates and Coupling Parameter

In both NRTidal v1 and NRTidalv2, the frequency-domain tidal phase correction $\Delta \phi_T(f)$ is parameterized by the dimensionless angular gravitational wave frequency:
$$M\omega = \pi M_{\rm tot} f$$
where $M_{\rm tot}$ is the total mass in seconds:
$$M_{\rm tot} = (m_1 + m_2) \frac{G M_\odot}{c^3} \approx (m_1 + m_2) \times 4.925491025543576 \times 10^{-6} \text{ s}$$
The PN expansion parameter is:
$$x = (M\omega)^{2/3} = (\pi M_{\rm tot} f)^{2/3}$$
The mass fractions are:
$$X_A = \frac{m_1}{m_1 + m_2}, \quad X_B = \frac{m_2}{m_1 + m_2}, \quad \eta = X_A X_B$$
The leading Newtonian tidal coupling parameter $\kappa_2^T$ is defined by:
$$\kappa_2^T = 2 \left[ X_B^4 X_A \Lambda_1 + X_A^4 X_B \Lambda_2 \right] = 2 \eta \left[ X_B^3 \Lambda_1 + X_A^3 \Lambda_2 \right]$$
The Newtonian prefactor is:
$$c_{\rm Newt} = \frac{39}{16} = 2.4375$$

---

### 1.2 NRTidal v1 Phase Model

The NRTidal v1 phase correction uses a $\text{Padé}(2|3)$ rational function:
$$\Delta \phi_T^{(v1)}(f) = -\frac{\kappa_2^T c_{\rm Newt}}{X_A X_B} x^{5/2} \frac{1 + n_1 x + n_{3/2} x^{3/2} + n_2 x^2 + n_{5/2} x^{5/2}}{1 + d_1 x + d_{3/2} x^{3/2}}$$

#### Exact IEEE FP64 Numerical Coefficients:
$$\begin{aligned}
c_{\rm Newt} &= 2.4375 \\
n_1 &= -17.428 \\
n_{3/2} &= 31.867 \\
n_2 &= -26.414 \\
n_{5/2} &= 62.362 \\
d_{3/2} &= 36.089
\end{aligned}$$

#### The $d_1$ Discrepancy:
In published literature (Dietrich et al. 2017 Eq. 7), the 1PN coefficient was constrained to match the analytic PN value:
$$d_1^{\rm analytic} = n_1 - \frac{3115}{1248} = -17.428 - 2.4959935897435896\dots = -19.92399358974359\dots$$
However, in the reference implementation (`LALSimNRTunedTides.c:325`), $d_1$ is hardcoded as:
$$d_1^{\rm LAL} = n_1 - 2.496 = -17.428 - 2.496 = -19.924$$
This difference of $\Delta d_1 \approx 6.41 \times 10^{-6}$ introduces a non-negligible phase accumulation across hundreds of cycles in the BNS band, explaining why earlier ports adhering strictly to the rational fraction $3115/1248$ failed exact numerical parity against LALSuite.

---

### 1.3 NRTidalv2 Phase Model

The NRTidalv2 model extends the rational function to order $\text{Padé}(3|2)$:
$$\Delta \phi_T^{(v2)}(f) = -\frac{\kappa_2^T c_{\rm Newt}}{X_A X_B} x^{5/2} \frac{1 + n_1 x + n_{3/2} x^{3/2} + n_2 x^2 + n_{5/2} x^{5/2} + n_3 x^3}{1 + d_1 x + d_{3/2} x^{3/2} + d_2 x^2}$$

#### Full Unrounded IEEE FP64 Coefficients:
$$\begin{aligned}
c_{\rm Newt} &= 2.4375 \\
n_1 &= -12.615214237993088 \\
n_{3/2} &= 19.0537346970349 \\
n_2 &= -21.166863146081035 \\
n_{5/2} &= 90.55082156324926 \\
n_3 &= -60.25357801943598 \\
d_1 &= -15.111207827736678 \\
d_{3/2} &= 22.195327350624694 \\
d_2 &= 8.064109635305156
\end{aligned}$$

*Note*: Literature publications (e.g. Dietrich et al. 2019 Table I) report rounded numbers ($n_{5/2} = 90.550822$, $n_3 = -60.253578$, $d_1 = -15.111208$, $d_2 = 8.0641096$). High-precision parity requires the full double-precision constants listed above.

---

### 1.4 NRTidalv2 Amplitude Correction

NRTidalv2 incorporates a planar tidal amplitude correction:
$$A_T(f) = -9 \kappa_2^T x^{13/4} \frac{1 + n_1 x + n_{2.89} x^{2.89}}{1 + d x^4}$$

#### Exact Coefficients:
$$\begin{aligned}
n_1 &= \frac{449}{108} \approx 4.157407407407407 \\
n_{2.89} &= \frac{22672}{9} \approx 2519.111111111111 \\
d &= 13477.8073677
\end{aligned}$$
*(The published paper quoted $d = 13477.8$, whereas $13477.8073677$ is the exact double-precision fit value).*

#### Exact Embedding in IMRPhenomD:
In `LALSimIMRPhenomD.c`, the baseline point-mass amplitude prefactor is:
$$A_0 = \frac{2 M^2}{D} \sqrt{\frac{5}{64\pi}}$$
When NRTidalv2 is evaluated within PhenomD, the tidal amplitude is added directly to the dimensionless PhenomD amplitude $A_{\rm PhenomD}(Mf)$:
$$A_{\rm net}(f) = A_0 \left[ A_{\rm PhenomD}(Mf) + 2 \sqrt{\frac{\pi}{5}} A_T(f) \right]$$
Notice the identity:
$$A_0 \cdot 2 \sqrt{\frac{\pi}{5}} = \frac{2 M^2}{D} \sqrt{\frac{5}{64\pi}} \cdot 2 \sqrt{\frac{\pi}{5}} = \frac{M^2}{2 D}$$
In `IMRPhenomD_NRTidal.c`, because `XLALSimIMRPhenomDGenerateFD` already includes this correction internally, `IMRPhenomD_NRTidal_Core` does **not** add any further amplitude correction; it only applies the high-frequency Planck taper $w_{\rm taper}(f)$.

---

### 1.5 Phase Anchoring and Reference Frequency Conventions in PhenomD_NRTidal

In `SimIMRPhenomD_NRTidal` and `SimIMRPhenomD_NRTidalv2`:
1. **Underlying PhenomD Phase**:
   The point-mass PhenomD phase $\phi_{\rm D}(f)$ is explicitly anchored at $f_{\rm ref}$ such that:
   $$\phi_{\rm D}(f) = \phi_{\rm IMR}(Mf) - t_0 (Mf - Mf_{\rm ref}) - [2 \phi_{\rm ref} + \phi_{\rm IMR}(Mf_{\rm ref})]$$
   where $t_0 = \phi_{\rm IMR}'(M f_{\rm peak})$. Consequently, $\phi_{\rm D}(f_{\rm ref}) = -2 \phi_{\rm ref}$.
2. **Tidal Phasing Wrapper**:
   In `IMRPhenomD_NRTidal_Core`, the frequency series is multiplied by the complex correction:
   $$\text{Corr}(f) = w_{\rm taper}(f) \exp\left[ -i \Delta \phi_T(f) - i \cdot {\rm pn\_fac} \cdot (SS_{3.5\rm PN} + SSS_{3.5\rm PN}) f^{2/3} \right]$$
   where:
   $${\rm pn\_fac} = \frac{3 (\pi M_{\rm tot})^{2/3}}{128 \eta}$$
3. **Absence of Reference Phase Subtraction**:
   - $\Delta \phi_T(f)$ is **never subtracted at $f_{\rm ref}$**. There is no $\Delta \phi_T(f_{\rm ref})$ offset subtraction.
   - The 3.5PN self-spin term is **never subtracted at $f_{\rm ref}$**.
   - No additional time shift is applied to the tidal phase.
   - Therefore, at $f = f_{\rm ref}$, the total phase has an offset:
     $$\phi_{\rm total}(f_{\rm ref}) = -2\phi_{\rm ref} - \Delta \phi_T(f_{\rm ref}) - {\rm pn\_fac} \cdot (SS_{3.5\rm PN} + SSS_{3.5\rm PN}) f_{\rm ref}^{2/3}$$
4. **Separation of Self-Spin Orders**:
   - The 2PN and 3PN self-spin (spin-induced quadrupole-monopole) terms are **not** applied in the tidal exponential wrapper.
   - Instead, when $\Lambda_{1,2} > 0$, the universal relation $Q(\Lambda)$ computes $d{\rm QuadMon}_i = Q(\Lambda_i) - 1$, which is passed directly into the TaylorF2 baseline inside PhenomD.
   - Because they reside inside the baseline PhenomD phase, the 2PN and 3PN self-spin contributions **are** anchored at $f_{\rm ref}$.
   - Only the 3.5PN tail terms ($SS_{3.5\rm PN}$ and $SSS_{3.5\rm PN}$) are applied in the external wrapper without $f_{\rm ref}$ subtraction.

---

## 2. IMRPhenomNSBH Waveform Model

### 2.1 Remnant Spin Parameterization

While IMRPhenomD uses the BBH remnant spin fit:
$$\chi_f^{\rm BBH} = {\rm FinalSpin0815}(\eta, \chi_{\rm BH}, \chi_{\rm NS})$$
IMRPhenomNSBH computes the remnant black hole spin $\chi_f^{\rm NSBH}$ from a dedicated NSBH numerical relativity remnant fit (incorporating tidal disruption and torus mass):
$$\chi_f = \chi_f^{\rm NSBH}(m_{\rm BH}, m_{\rm NS}, \chi_{\rm BH}, \Lambda_{\rm NS})$$
The PhenomD phase connection coefficients (`pPhi`) and intermediate collocation coefficients are computed using $\chi_f^{\rm NSBH}$. This alters the intermediate and ringdown phase evolution relative to a BBH baseline.

---

### 2.2 Inspiral Amplitude Formulation

In `LALSimIMRPhenomNSBH.c`, the inspiral amplitude $a_{\rm PN}(f)$ is computed using the **Stationary Phase Approximation (SPA)** derived from the time-domain 3.5PN expansion of IMRPhenomC (Santamaría et al. 2010, arXiv:1005.3306):
$$a_{\rm PN}(f) = \sqrt{|{\rm AmpPN}_{\rm re}|^2 + |{\rm AmpPN}_{\rm im}|^2}$$
where $v = (\pi M_{\rm tot} f)^{1/3}$ is the orbital velocity parameter.

#### 1. Orbital frequency derivative $\dot{\omega}$:
$$\dot{\omega} = \frac{3}{2} v \dot{x}$$
where the dimensionless energy flux parameter derivative $\dot{x}$ is evaluated to 3.5PN order (TaylorT4 / Blanchet):
$$\dot{x} = \dot{x}_N v^{10} \left[ 1 + \dot{x}_2 v^2 + \dot{x}_3 v^3 + \dot{x}_4 v^4 + \dot{x}_5 v^5 + (\dot{x}_6 + \dot{x}_{6\rm log} \log v^2) v^6 + \dot{x}_7 v^7 \right]$$
with leading-order factor $\dot{x}_N = \frac{64\eta}{5}$.

#### 2. Time-domain amplitude components:
$$A_{\rm TD}(v) = A_N v^2 \left[ A_{\rm real}(v) + i A_{\rm imag}(v) \right]$$
where $A_N = 8\eta \sqrt{\frac{\pi}{5}}$, and:
$$\begin{aligned}
A_{\rm real}(v) &= 1 + A_2 v^2 + A_3 v^3 + A_4 v^4 + A_5 v^5 + (A_6 + A_{6\rm log} \log v^2) v^6 \\
A_{\rm imag}(v) &= A_{5\rm imag} v^5 + A_{6\rm imag} v^6
\end{aligned}$$
with tail coefficients:
$$A_{5\rm imag} = -24\eta, \quad A_{6\rm imag} = \frac{428\pi}{105}$$

#### 3. SPA amplitude factor:
$${\rm ampfac} = \sqrt{\left| \frac{\pi}{\dot{\omega}} \right|}$$
$$\begin{aligned}
{\rm AmpPN}_{\rm re} &= {\rm ampfac} \cdot A_N v^2 A_{\rm real}(v) \\
{\rm AmpPN}_{\rm im} &= {\rm ampfac} \cdot A_N v^2 A_{\rm imag}(v)
\end{aligned}$$

#### 4. Exact rational coefficients in IMRPhenomC:
$$\begin{aligned}
A_2 &= \frac{-107 + 55\eta}{42} \\
A_3 &= 2\pi - \frac{4\chi}{3} + \frac{2\eta (2\chi)}{3} \\
A_4 &= -\frac{2173}{1512} - \eta \left( \frac{1069}{216} - 2\chi^2 \right) + \frac{2047\eta^2}{1512} \\
A_5 &= -\frac{107\pi}{21} + \frac{34\pi\eta}{21} \\
A_{6\rm log} &= -\frac{428}{105} \\
A_6 &= \frac{27027409}{646800} - \frac{856\gamma_E}{105} + \frac{2\pi^2}{3} + \eta\left(\frac{41\pi^2}{96} - \frac{278185}{33264}\right) - \frac{20261\eta^2}{2772} + \frac{114635\eta^3}{99792} - \frac{428\log 16}{105}
\end{aligned}$$

#### 5. Windowed combination with merger-ringdown:
$$\text{amp}(f) = -\left[ a_{\rm PN} w^-(f, \epsilon_{\rm ins} \tilde{f}_0^{\rm PN}, d_0 + \sigma_{\rm tide}) + a_{\rm PM} w^-(f, \tilde{f}_0^{\rm PM}, d_0 + \sigma_{\rm tide}) + a_{\rm RD} w^+(f, \tilde{f}_0^{\rm RD}, d_0 + \sigma_{\rm tide}) \right]$$
where the hyperbolic-tangent transition windows are:
$$w^-(f, f_0, d) = \frac{1}{2}\left[ 1 - \tanh\left(\frac{4(Mf - Mf_0)}{d}\right) \right], \quad w^+(f, f_0, d) = \frac{1}{2}\left[ 1 + \tanh\left(\frac{4(Mf - Mf_0)}{d}\right) \right]$$

---

### 2.3 Exact Formulas for Time Shift $t_0$ and Reference Phase $\phi_{\rm precalc}$

#### 1. Reference Phase Precalculation $\phi_{\rm precalc}$:
At reference frequency $f_{\rm ref}$, the baseline PhenomD phase is evaluated at $Mf_{\rm ref} = M_{\rm sec} f_{\rm ref}$:
$$\phi_0 = \phi_{\rm PhenomD}^{\rm NSBH}(Mf_{\rm ref})$$
Because $\phi_{\rm ref}$ represents orbital phase, a factor of 2 is applied:
$$\phi_{\rm precalc} = 2\phi_{\rm ref} + \phi_0 = 2\phi_{\rm ref} + \phi_{\rm PhenomD}^{\rm NSBH}(M_{\rm sec} f_{\rm ref})$$

#### 2. Exact Formula for Time Shift $t_0$:
In `LALSimIMRPhenomNSBH.c:210`, $t_0$ is computed via `DPhiMRD(f_max * M_sec, pPhi, 1.0, 1.0)`:
$$t_0^{\rm NSBH} = \left. \frac{d\phi_{\rm MRD}}{df}\right|_{f = f_{\rm max} M_{\rm sec}} = \frac{1}{\eta} \left[ \alpha_1 + \frac{\alpha_2}{f_{\rm eval}^2} + \frac{\alpha_3}{f_{\rm eval}^{1/4}} + \frac{\alpha_4}{f_{\rm damp} \left( 1 + \frac{(f_{\rm eval} - \alpha_5 f_{\rm RD})^2}{f_{\rm damp}^2} \right)} \right]$$
evaluated at:
$$f_{\rm eval} = M_{\rm sec} f_{\rm max}$$
*(If $f_{\rm max} = 0$, it defaults to $f_{\rm eval} = 0.2$).*

#### 3. Total Phase Assembly:
For each frequency $f$:
$$\phi(f) = \phi_{\rm PhenomD}^{\rm NSBH}(Mf) + \Delta \phi_T(f) - t_0^{\rm NSBH} (Mf - Mf_{\rm ref}) - \phi_{\rm precalc}$$
where $\Delta \phi_T(f)$ is the NRTidalv2 phase evaluated with $\Lambda_{\rm BH} = 0$ and $\Lambda_{\rm NS} > 0$.
Notice that at $f = f_{\rm ref}$:
$$\phi(f_{\rm ref}) = \phi_0 + \Delta \phi_T(f_{\rm ref}) - 0 - (2\phi_{\rm ref} + \phi_0) = \Delta \phi_T(f_{\rm ref}) - 2\phi_{\rm ref}$$
The physical strain is then:
$$\tilde{h}(f) = A_0 \cdot \text{amp}(f) \cdot e^{-i \phi(f)} \cdot \left[ 2 \sqrt{\frac{5}{64\pi}} \right]$$
where $A_0 = \frac{M^2}{D}$.

---

## 3. IMRPhenomXHM Multimode Alignment

IMRPhenomXHM models higher-order spherical harmonic modes $(\ell, |m|) \in \{(2,2), (2,1), (3,3), (3,2), (4,4)\}$.

### 3.1 Mode (2,2) Reference Shifts

The dominant (2,2) mode establishes the time and phase reference:
1. **Time Shift $t_{\rm shift}^{22}$**:
   $$t_{\rm shift}^{22} = b_{\rm fit}(\eta, S_{\rm tot}, \delta\chi, \delta) - \left. \frac{1}{\eta} \frac{d\phi_{22}}{d(Mf)} \right|_{f_{\rm ref}^{\rm fit}} - 2\pi (500 + \Delta t_{\psi_4 \to h})$$
   where $f_{\rm ref}^{\rm fit} = f_{\rm RING}^{22} - f_{\rm DAMP}^{22}$, and $\Delta t_{\psi_4 \to h}$ is the time offset between the peak of $\psi_4$ and the peak of the strain $h$.
2. **Reference Phase $\phi_{\rm ref}^{22}$**:
   $$\phi_{\rm ref}^{22} = -\left[ \frac{1}{\eta} \phi_{22}(Mf_{\rm ref}) + t_{\rm shift}^{22} Mf_{\rm ref} \right] + 2\phi_0 + \frac{\pi}{4}$$

---

### 3.2 Alignment Frequency $f_{\rm align}$

For higher-order modes $(\ell, m)$, the inspiral phase is calibrated to the (2,2) phase at an alignment frequency $f_{\rm align}$:
$$f_{\rm align} = \begin{cases}
0.6 \cdot \frac{m}{2} \cdot f_{\rm MECO}, & \text{if } \eta > \eta_{\rm EMR} = 0.05 \\
\frac{m}{2} \cdot f_{\rm MECO}, & \text{if } \eta \le 0.05
\end{cases}$$
where $f_{\rm MECO}$ is the Minimum Energy Circular Orbit frequency of the system.

---

### 3.3 Mode-Specific Phase Offset $\Delta \phi_{\ell m}$

At $f_{\rm align}$, the mode phase is aligned to match the post-Newtonian scaling relation:
$$\phi_{\ell m}(f) \approx \frac{m}{2} \phi_{22}\left(\frac{2}{m} f\right) - \frac{3\pi}{4} \left(1 - \frac{m}{2}\right)$$
The exact phase offset $\Delta \phi_{\ell m}$ added to the mode phase is:
$$\Delta \phi_{\ell m} = \left[ \frac{m}{2} \left( \frac{1}{\eta} \phi_{22}\left(\frac{2}{m} f_{\rm align}\right) + \phi_{\rm ref}^{22} \right) + t_{\rm shift}^{22} f_{\rm align} - \frac{3\pi}{4}\left(1 - \frac{m}{2}\right) - \phi_{\ell m}^{\rm raw}(f_{\rm align}) \right] \bmod 2\pi$$
where:
$$\phi_{\ell m}^{\rm raw}(f_{\rm align}) = \phi_{\ell m}^{\rm insp}(f_{\rm align}) + C_{1}^{\rm INSP} f_{\rm align} + C^{\rm INSP}$$

---

### 3.4 Mode (2,1) Amplitude Sign Rule

The real part of the leading-order PN amplitude for mode (2,1) is evaluated at $ff = 0.008$:
$$\begin{aligned}
\mathcal{A}_{21}^{\rm PN}(ff) &= -\frac{16 \delta \eta ff \pi^{3/2}}{3\sqrt{5}} \\
&\quad + \frac{4 \cdot 2^{1/3} \left[(\chi_1 - \chi_2) + \delta (\chi_1 + \chi_2)\right] \eta ff^{4/3} \pi^{11/6}}{\sqrt{5}} \\
&\quad + \frac{2 \cdot 2^{2/3} \eta (306\delta - 360\delta\eta) ff^{5/3} \pi^{13/6}}{189\sqrt{5}}
\end{aligned}$$
Because the amplitude in PhenomXHM is defined to be positive-definite:
- If $\mathcal{A}_{21}^{\rm PN}(0.008) \ge 0$, the sign has flipped relative to the standard convention, requiring:
  $$\Delta \phi_{21} \gets \Delta \phi_{21} + \pi$$
- Else, no additional phase shift is applied.

---

### 3.5 Mode (3,2) Spheroidal-Spherical Mixing

Mode (3,2) undergoes significant mode mixing with (2,2) in the ringdown regime (`MixingOn = 1`).
1. **Spheroidal Time Shift**:
   At $f_{\rm ref}^{\rm RD} = f_{\rm RING}^{22} + f_{\rm DAMP}^{22}$:
   $$t_{\rm shift}^{32} = {\rm Fit\_TimeShift}_{32}(\eta, S_{\rm tot}, \chi_{1z}, \chi_{2z})$$
   $$d\phi_{22}^{\rm ref} = \frac{1}{\eta} \phi_{22}'(f_{\rm ref}^{\rm RD}) + t_{\rm shift}^{22}$$
   The spheroidal ringdown phase derivative constant $\alpha_{0, S}$ is shifted by:
   $$\alpha_{0, S} \gets \alpha_{0, S} + d\phi_{22}^{\rm ref} + t_{\rm shift}^{32} - \phi_{32, \rm spheroidal}'(f_{\rm ref}^{\rm RD})$$
2. **Spheroidal Phase Shift**:
   At $f_{\rm ref}^{\rm RD} = f_{\rm RING}^{22}$:
   $$\phi_{\rm shift}^{32} = {\rm Fit\_PhaseShift}_{32}(\eta, S_{\rm tot}, \chi_{1z}, \chi_{2z})$$
   $$\phi_{22}^{\rm ref} = \frac{1}{\eta} \phi_{22}(f_{\rm ref}^{\rm RD}) + t_{\rm shift}^{22} f_{\rm ref}^{\rm RD} + \phi_{\rm ref}^{22}$$
   $$\phi_{0, S} = \phi_{22}^{\rm ref} - \phi_{32, \rm spheroidal}(f_{\rm ref}^{\rm RD}) + \phi_{\rm shift}^{32}$$
3. The intermediate region coefficients are matched to the rotated spheroidal-to-spherical waveform at the boundary $f_{\rm match}^{\rm IM}$.
4. Finally, the global offset $\Delta \phi_{32}$ is applied at $f_{\rm align}$ via the general formula in Section 3.3.

---

## 4. IMRPhenomXP / XPHM Precession (Convention 1)

In precessing approximants, the coprecessing-frame waveform $h_{\ell m}^{\rm coprec}(f)$ is rotated into the inertial observer frame via time-dependent or frequency-dependent Euler angles $(\alpha(f), \beta(f), \epsilon(f))$.

### 4.1 Source Frame Geometry

At reference frequency $f_{\rm ref}$, the source frame (SF) is defined such that $\hat{L}(f_{\rm ref}) = \hat{z}_{\rm SF}$.
1. **Orbital Angular Momentum**:
   $$\vec{L}_{\rm ref} = (0, 0, L_{\rm ref})$$
   where $L_{\rm ref} = M_{\rm tot}^2 L_{\rm PN}(v_{\rm ref})$ and $v_{\rm ref} = (\pi M_{\rm tot} f_{\rm ref})^{1/3}$.
2. **Total Angular Momentum $\vec{J}_0$**:
   $$\vec{J}_0 = \vec{L}_{\rm ref} + m_1^2 \vec{\chi}_1 + m_2^2 \vec{\chi}_2 = (J_{0x}^{\rm SF}, J_{0y}^{\rm SF}, J_{0z}^{\rm SF})$$
   $$J_0 = |\vec{J}_0|$$
3. **Angles of $\vec{J}_0$ in SF**:
   $$\theta_J^{\rm SF} = \arccos\left(\frac{J_{0z}^{\rm SF}}{J_0}\right), \quad \phi_J^{\rm SF} = \operatorname{atan2}(J_{0y}^{\rm SF}, J_{0x}^{\rm SF})$$
   (If $J_{0x}^{\rm SF} = J_{0y}^{\rm SF} = 0$, then $\phi_J^{\rm SF} = 0$).

---

### 4.2 Line-of-Sight Vector and Angle $\kappa$

The line-of-sight unit vector $\vec{N}$ in the source frame is:
$$\vec{N}_{\rm SF} = \begin{pmatrix} \sin\iota \sin\phi_{\rm ref} \\ \sin\iota \cos\phi_{\rm ref} \\ \cos\iota \end{pmatrix}$$
Rotating $\vec{N}_{\rm SF}$ to align $\vec{J}_0$ with the $z$-axis via $R_y(-\theta_J^{\rm SF}) R_z(-\phi_J^{\rm SF})$:
$$\vec{N}' = R_y(-\theta_J^{\rm SF}) R_z(-\phi_J^{\rm SF}) \vec{N}_{\rm SF}$$
The rotation angle $\kappa$ bringing $\vec{N}'$ into the $(+x, z)$ half-plane of the $J$ frame is:
$$\kappa = \operatorname{atan2}(N'_y, N'_x)$$

---

### 4.3 Reference Euler Angles in Convention 1 (Default)

In Convention 1:
$$\begin{aligned}
\alpha_0 &= \pi - \kappa \\
\epsilon_0 &= \phi_J^{\rm SF} - \pi \\
\theta_{JN} &= \arccos\left(\frac{\vec{J}_0 \cdot \vec{N}_{\rm SF}}{J_0}\right) \\
\phi_0^{\rm aligned} &= 0
\end{aligned}$$
For $\beta(f)$:
$$\cos\beta(f) = \frac{J^2(f) + L^2(f) - S^2(f)}{2 J(f) L(f)}$$
There is **no** offset subtraction for $\beta$. At $f_{\rm ref}$, $\beta(f_{\rm ref}) = \arccos(\cos\beta(f_{\rm ref}))$.

---

### 4.4 Reference Frequency Subtractions

To ensure that at $f = f_{\rm ref}$, the Euler angles match $(\alpha_0, \beta(f_{\rm ref}), \epsilon_0)$, the evaluated angles $\alpha(f)$ and $\epsilon(f)$ are shifted by:
$$\begin{aligned}
\alpha_{\rm offset} &= \alpha_{\rm raw}(f_{\rm ref}) - \alpha_0 \\
\epsilon_{\rm offset} &= \epsilon_{\rm raw}(f_{\rm ref}) - \epsilon_0
\end{aligned}$$
Thus, at any frequency $f$:
$$\begin{aligned}
\alpha(f) &= \alpha_{\rm raw}(f) - \alpha_{\rm offset} = \alpha_0 + [\alpha_{\rm raw}(f) - \alpha_{\rm raw}(f_{\rm ref})] \\
\epsilon(f) &= \epsilon_{\rm raw}(f) - \epsilon_{\rm offset} = \epsilon_0 + [\epsilon_{\rm raw}(f) - \epsilon_{\rm raw}(f_{\rm ref})]
\end{aligned}$$

---

### 4.5 Polarization Rotation Angle $\zeta$

The waveframe unit vector $\vec{X}_{\rm SF}$ orthogonal to $\vec{N}_{\rm SF}$ is:
$$\vec{X}_{\rm SF} = \begin{pmatrix} -\cos\iota \sin\phi_{\rm ref} \\ -\cos\iota \cos\phi_{\rm ref} \\ \sin\iota \end{pmatrix}$$
Rotating $\vec{X}_{\rm SF}$ into the $J$ frame via $R_z(-\kappa) R_y(-\theta_J^{\rm SF}) R_z(-\phi_J^{\rm SF})$:
$$\vec{X}_{\rm JF} = R_z(-\kappa) R_y(-\theta_J^{\rm SF}) R_z(-\phi_J^{\rm SF}) \vec{X}_{\rm SF}$$
In Convention 1, the Arun et al. basis vectors $(\vec{P}, \vec{Q})$ in the $J$ frame are:
$$\vec{P}_{\rm Arun} = \begin{pmatrix} \cos\theta_{JN} \\ 0 \\ -\sin\theta_{JN} \end{pmatrix}, \quad \vec{Q}_{\rm Arun} = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}$$
The projections are:
$$\begin{aligned}
X \cdot P &= X_{x}^{\rm JF} \cos\theta_{JN} - X_{z}^{\rm JF} \sin\theta_{JN} \\
X \cdot Q &= X_{y}^{\rm JF}
\end{aligned}$$
The polarization rotation angle $\zeta$ is:
$$\zeta = \operatorname{atan2}(X \cdot Q, X \cdot P)$$

---

### 4.6 Polarization Rotation Matrix

To transform the coprecessing-frame metric projection into the detector radiation frame:
$$\begin{pmatrix} h_+ \\ h_\times \end{pmatrix} = \begin{pmatrix} \cos(2\zeta) & \sin(2\zeta) \\ -\sin(2\zeta) & \cos(2\zeta) \end{pmatrix} \begin{pmatrix} h_+^0 \\ h_\times^0 \end{pmatrix}$$
In complex strain notation:
$$h_+ + i h_\times = e^{-2i\zeta} (h_+^0 + i h_\times^0)$$
$$h_+ - i h_\times = e^{2i\zeta} (h_+^0 - i h_\times^0)$$

---

## 5. Summary Checklist for Implementation

| Model | Quantity | Exact Formula / Source | Parity Trap Identified |
|---|---|---|---|
| **NRTidal v1** | $d_1$ coefficient | $n_1 - 2.496 = -19.924$ | Using analytical $3115/1248$ creates a phase drift |
| **NRTidalv2** | Padé coefficients | 9 unrounded FP64 constants in Table 1.3 | Using 6-8 digit rounded coefficients creates $O(10^{-4})$ mismatch |
| **NRTidalv2** | Amplitude $A_T$ | $A_{\rm total} = A_0 (A_{\rm BBH} + 2\sqrt{\pi/5} A_T)$ | Missing $2\sqrt{\pi/5}$ factor or double-applying $A_T$ in wrapper |
| **NRTidalv2** | Phase Anchoring | No subtraction of $\Delta \phi_T(f_{\rm ref})$ or 3.5PN self-spin | Subtracting $\Delta \phi_T(f_{\rm ref})$ destroys reference phase parity |
| **PhenomNSBH** | Time shift $t_0$ | Evaluated at $Mf = M_{\rm sec} f_{\rm max}$ (grid bound) | PhenomD evaluates at $Mf_{\rm peak}$; NSBH depends on grid $f_{\rm max}$ |
| **PhenomNSBH** | Remnant spin | $\chi_f^{\rm NSBH}$ from remnant fits | Using BBH `FinalSpin0815` alters intermediate coefficients |
| **PhenomXHM** | Mode phase offset $\Delta \phi_{\ell m}$ | Evaluated at $f_{\rm align} = 0.6 \frac{m}{2} f_{\rm MECO}$ via Eq. 4.13 | Omitting $(2,1)$ amplitude sign check flips $(2,1)$ by $\pi$ |
| **PhenomXP** | Euler offsets | $\alpha_0 = \pi - \kappa, \epsilon_0 = \phi_J - \pi$ | Convention 1 zeroes $\phi_0^{\rm aligned}$ and sets $\epsilon_0 = \phi_J - \pi$ |
| **PhenomXP** | Polarization | $\begin{pmatrix} \cos 2\zeta & \sin 2\zeta \\ -\sin 2\zeta & \cos 2\zeta \end{pmatrix}$ | Sign error in $X \cdot P$ or $\zeta$ rotation matrix |
