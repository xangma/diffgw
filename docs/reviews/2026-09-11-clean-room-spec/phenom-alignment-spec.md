# Clean-Room Mathematical Specification: Phenom Reference Phase & Time Alignment

## 1. Executive Summary & Clean-Room Boundary Rules

This document establishes the exact mathematical specification for reference frequency ($f_{\rm ref}$), reference phase ($\phi_0$), and reference time ($t_0$) alignment across the aligned-spin phenomenological waveform families **IMRPhenomD** and **IMRPhenomXAS**.

### Strict Clean-Room Rules
- Output consists purely of raw mathematical facts, analytical expressions, and unrounded IEEE FP64 constants.
- No code, control structures, macros, or typedefs are copied or transplanted from external sources.
- Governed under pure MIT license integrity.
- Primary literature references:
  - **Khan et al. (2016)**, Phys. Rev. D 93, 044007 [arXiv:1508.07253] (Paper II: PhenomD IMR)
  - **Husa et al. (2016)**, Phys. Rev. D 93, 044006 [arXiv:1508.07250] (Paper I: PhenomD Inspiral)
  - **Pratten et al. (2020)**, Phys. Rev. D 102, 064001 [arXiv:2001.10914] (PhenomXAS)
  - **Pratten et al. (2020)**, Phys. Rev. D 103, 104056 [arXiv:2004.06503] (PhenomXHM)

---

## 2. Fundamental Frequency & Mass Definitions

Let:
- $m_1, m_2$: Component masses in SI units ($m_1 \ge m_2$).
- $M = m_1 + m_2$: Total mass.
- $q = m_1 / m_2 \ge 1$: Mass ratio.
- $\eta = \frac{m_1 m_2}{M^2} = \frac{q}{(1+q)^2}$: Symmetric mass ratio.
- $M_{\rm sec} = \frac{G M}{c^3} = M_\odot \left(\frac{M}{M_\odot}\right) \frac{G}{c^3}$: Total mass converted to geometric time (seconds).
- $Mf = M_{\rm sec} \cdot f$: Dimensionless geometric frequency.
- $Mf_{\rm ref} = M_{\rm sec} \cdot f_{\rm ref}$: Dimensionless reference frequency.
- $\chi_1, \chi_2 \in [-1, 1]$: Dimensionless aligned spins along $\hat{L}$.
- $\chi_{\rm PN}$: Post-Newtonian reduced spin parameter:
  $$\chi_{\rm PN} = \frac{1}{2}\left( (\chi_1 + \chi_2) \left(1 - \frac{76}{113}\eta\right) + \sqrt{1 - 4\eta}(\chi_1 - \chi_2) \right)$$
- $\xi = \chi_{\rm PN} - 1$: Shifted reduced spin parameter used in fitting polynomials.

---

## 3. IMRPhenomD Reference Alignment Specification

### 3.1 Frequency-Domain Waveform Convention
In IMRPhenomD, the complex frequency-domain strain is defined with a negative phase sign in the exponential:
$$\tilde{h}(f) = A(f) \exp\left( -i \phi_{\rm total}(f) \right)$$

### 3.2 Peak Frequency Determination ($Mf_{\rm peak}$)
The merger-ringdown amplitude ansatz is given by:
$$A_{\rm MRD}(Mf) = \exp\left( -\frac{(Mf - f_{\rm RD})\gamma_2}{f_{\rm DM}\gamma_3} \right) \frac{f_{\rm DM}\gamma_3 \gamma_1}{(Mf - f_{\rm RD})^2 + (f_{\rm DM}\gamma_3)^2}$$
where $f_{\rm RD}$ is the quasi-normal mode (QNM) ringdown frequency and $f_{\rm DM}$ is the ringdown damping frequency.

The analytical location of the maximum of $A_{\rm MRD}$, denoted $Mf_{\rm peak}$ (or $f_{\rm maxCalc}$ in LALSimulation), is derived from $\frac{d A_{\rm MRD}}{d(Mf)} = 0$ (Eq. 20 of Khan et al. 2016):
$$Mf_{\rm peak} = \begin{cases}
\left| f_{\rm RD} + \frac{f_{\rm DM} \left(-1 + \sqrt{1 - \gamma_2^2}\right)\gamma_3}{\gamma_2} \right|, & \gamma_2 \le 1 \\
\left| f_{\rm RD} - \frac{f_{\rm DM} \gamma_3}{\gamma_2} \right|, & \gamma_2 > 1
\end{cases}$$

### 3.3 Reference Time Offset ($t_0$)
To ensure that the amplitude peak occurs near $t = 0$ in the time domain, the stationary phase condition requires:
$$t(f) = \frac{1}{2\pi} \frac{d\phi}{df} = 0 \quad \text{at } Mf = Mf_{\rm peak}$$

In IMRPhenomD, the dimensionless derivative of the merger-ringdown phase ansatz is:
$$\left. \frac{d\Phi_{\rm MRD}^{\rm Ansatz}}{d(Mf)} \right|_{Mf} = \frac{1}{\eta} \left[ \alpha_1 + \frac{\alpha_2}{(Mf)^2} + \frac{\alpha_3}{(Mf)^{1/4}} + \frac{\alpha_4}{f_{\rm DM} \left(1 + \left(\frac{Mf - \alpha_5 f_{\rm RD}}{f_{\rm DM}}\right)^2\right)} \right]$$

The dimensionless time offset parameter $t_0$ is defined evaluated directly at $Mf_{\rm peak}$:
$$t_0 = \left. \frac{d\Phi_{\rm MRD}^{\rm Ansatz}}{d(Mf)} \right|_{Mf = Mf_{\rm peak}}$$
*(Note: $t_0$ is evaluated from the base ansatz derivative without the intermediate-to-ringdown $C_2^{\rm MRD}$ connection shift).*

In physical units of seconds, this corresponds to:
$$\Delta t_0 = M_{\rm sec} \cdot t_0$$

### 3.4 Phase Alignment at Reference Frequency
Let $\phi_{\rm raw}(Mf)$ denote the piece-wise $C^1$-continuous IMRPhenomD raw phase:
$$\phi_{\rm raw}(Mf) = \begin{cases}
\Phi_{\rm ins}(Mf), & Mf \le 0.018 \\
\frac{1}{\eta}\Phi_{\rm int}(Mf) + C_1^{\rm int} + C_2^{\rm int} (Mf), & 0.018 < Mf < 0.5 f_{\rm RD} \\
\frac{1}{\eta}\Phi_{\rm MRD}(Mf) + C_1^{\rm MRD} + C_2^{\rm MRD} (Mf), & Mf \ge 0.5 f_{\rm RD}
\end{cases}$$
where $C_1^{\rm int}, C_2^{\rm int}, C_1^{\rm MRD}, C_2^{\rm MRD}$ enforce $C^1$ continuity at $Mf = 0.018$ and $Mf = 0.5 f_{\rm RD}$.

The phase evaluated at the dimensionless reference frequency $Mf_{\rm ref} = M_{\rm sec} f_{\rm ref}$ is:
$$\phi_{\rm ref} = \phi_{\rm raw}(Mf_{\rm ref})$$

The precalculated reference phase shift is:
$$\phi_{\rm precalc} = 2 \phi_0 + \phi_{\rm ref}$$
*(where the factor of 2 accounts for $\phi_0$ being the orbital phase, while the dominant gravitational-wave mode is $(2, 2)$)*.

### 3.5 Final Waveform Phase
At any arbitrary frequency $f$, the adjusted phase subtracted inside the exponential is:
$$\phi_{\rm total}(Mf) = \phi_{\rm raw}(Mf) - t_0 (Mf - Mf_{\rm ref}) - \phi_{\rm precalc}$$
$$= \phi_{\rm raw}(Mf) - \phi_{\rm raw}(Mf_{\rm ref}) - t_0 M_{\rm sec} (f - f_{\rm ref}) - 2 \phi_0$$

Evaluating at $f = f_{\rm ref}$:
$$\phi_{\rm total}(Mf_{\rm ref}) = -2\phi_0$$
Thus, the complex strain evaluates at $f_{\rm ref}$ to:
$$\tilde{h}(f_{\rm ref}) = A(f_{\rm ref}) \exp\left(-i (-2\phi_0)\right) = A(f_{\rm ref}) \exp\left(+2 i \phi_0\right)$$

---

## 4. IMRPhenomXAS Reference Alignment Specification

### 4.1 Frequency-Domain Waveform Convention
In IMRPhenomXAS (and the entire PhenomX family), the complex frequency-domain strain is defined with a **positive** phase sign in the exponential:
$$\tilde{h}_{22}(f) = A_{22}(f) \exp\left( +i \phi_{22}(f) \right)$$

### 4.2 Time Shift Formulation ($t_{\rm shift}$ / `linb`)
In PhenomX, time alignment is calibrated against numerical relativity hybrid waveforms where the Newman-Penrose scalar $\psi_4$ peaks $500 M$ before the end of the waveform.

The time shift slope $t_{\rm shift}$ (denoted `linb` in LALSimulation) consists of three components:
$$t_{\rm shift} = \text{linb}_{\rm fit}(\eta, S, \delta\chi, \delta) - \left. \frac{1}{\eta} \frac{d\Phi_{22}}{d(Mf)} \right|_{f_{\rm fit}} - 2\pi (500 + \Delta t_{\psi_4 \to h})$$
where:
1. $f_{\rm fit} = f_{\rm RING} - f_{\rm DAMP}$: The calibration reference frequency.
2. $\left. \frac{1}{\eta} \frac{d\Phi_{22}}{d(Mf)} \right|_{f_{\rm fit}}$: The analytical derivative of the model's phase at $f_{\rm fit}$.
3. $\text{linb}_{\rm fit}(\eta, S, \delta\chi, \delta)$: An unrounded polynomial fit of $d\phi_{22}/d(Mf)$ at $f_{\rm fit}$ calibrated across the parameter space:
   $$\text{linb}_{\rm fit} = P_{\rm noSpin}(\eta) + P_{\rm eqSpin}(\eta, S) - 148.17317525117338 \cdot \delta\chi \cdot \delta \cdot \eta^2$$
   with $S = \frac{m_1^2 \chi_1 + m_2^2 \chi_2}{M^2}$, $\delta = \frac{m_1 - m_2}{M}$, $\delta\chi = \chi_1 - \chi_2$.
4. $\Delta t_{\psi_4 \to h} = \text{XLALSimIMRPhenomXPsi4ToStrain}(\eta, S, \delta\chi)$: Polynomial fit accounting for the time difference between the peak of $|\psi_4|$ and the peak of strain $|h|$:
   $$\Delta t_{\psi_4 \to h} = P_{\psi_4, \rm noSpin}(\eta) + P_{\psi_4, \rm eqSpin}(\eta, S) + 105.37711654943146 \cdot \delta\chi \sqrt{1 - 4\eta}\eta^2$$
5. $500$: Geometric duration from $\psi_4$ peak to waveform termination in hybrid data.

### 4.3 Reference Phase Offset ($\text{phifRef}$)
In PhenomXAS, the phase at reference frequency includes the standard Stationary Phase Approximation (SPA) $+ \pi/4$ phase shift:
$$\text{phifRef} = -\left( \phi_{\rm raw}(Mf_{\rm ref}) + t_{\rm shift} \cdot Mf_{\rm ref} \right) + 2\phi_0 + \frac{\pi}{4}$$
*(assuming vacuum non-tidal baseline)*.

### 4.4 Final Waveform Phase
For any frequency $f$, the phase added inside $\exp(+i \phi)$ is:
$$\phi_{22}(Mf) = \phi_{\rm raw}(Mf) + t_{\rm shift} \cdot Mf + \text{phifRef}$$
$$= \phi_{\rm raw}(Mf) - \phi_{\rm raw}(Mf_{\rm ref}) + t_{\rm shift} (Mf - Mf_{\rm ref}) + 2\phi_0 + \frac{\pi}{4}$$

Evaluating at $f = f_{\rm ref}$:
$$\phi_{22}(Mf_{\rm ref}) = 2\phi_0 + \frac{\pi}{4}$$
Thus, the complex strain evaluates at $f_{\rm ref}$ to:
$$\tilde{h}_{22}(f_{\rm ref}) = A_{22}(f_{\rm ref}) \exp\left( +i \left(2\phi_0 + \frac{\pi}{4}\right) \right)$$

---

## 5. Comparison & Cross-Model Translation

| Feature | IMRPhenomD | IMRPhenomXAS |
| :--- | :--- | :--- |
| **Fourier Exponential** | $\exp\left(-i \phi\right)$ | $\exp\left(+i \phi\right)$ |
| **Phase at $f_{\rm ref}$** | $\phi(f_{\rm ref}) = -2\phi_0$ (giving $\tilde{h} \propto e^{+2i\phi_0}$) | $\phi(f_{\rm ref}) = 2\phi_0 + \frac{\pi}{4}$ (giving $\tilde{h} \propto e^{i(2\phi_0 + \pi/4)}$) |
| **Time Alignment Point** | Derivative at $Mf_{\rm peak}$ of MRD amplitude | Calibration fit to hybrid NR $\psi_4$ peak + $\psi_4 \to h$ shift |
| **Sign of Time Term** | $- t_0 (Mf - Mf_{\rm ref})$ | $+ t_{\rm shift} (Mf - Mf_{\rm ref})$ |
| **Physical Sign** | Matches $-2\pi f \Delta t$ under $-i\phi$ | Matches $+2\pi f \Delta t$ under $+i\phi$ |

---

## 6. Table V Phenomenological Fit Definitions (Khan et al. 2016)

Every parameter $\Lambda \in \{\rho_1, \rho_2, \rho_3, v_2, \gamma_1, \gamma_2, \gamma_3, \sigma_1, \sigma_2, \sigma_3, \sigma_4, \beta_1, \beta_2, \beta_3, \alpha_1, \alpha_2, \alpha_3, \alpha_4, \alpha_5\}$ is evaluated using the 11-coefficient hierarchical ansatz (Eq. 31 of Khan et al. 2016):

$$\Lambda(\eta, \chi_{\rm PN}) = \lambda_{00} + \lambda_{10} \eta + \xi \left( \lambda_{01} + \lambda_{11} \eta + \lambda_{21} \eta^2 \right) + \xi^2 \left( \lambda_{02} + \lambda_{12} \eta + \lambda_{22} \eta^2 \right) + \xi^3 \left( \lambda_{03} + \lambda_{13} \eta + \lambda_{23} \eta^2 \right)$$

where $\xi = \chi_{\rm PN} - 1$.

### 6.1 Inspiral Amplitude Coefficients
- **$\rho_1$** (Eq. B4):
  $$\lambda = (3931.8979897196696, -17395.758706812805, 3132.375545898835, 343965.86092361377, -1.2162565819981997\times 10^6, -70698.00600428853, 1.383907177859705\times 10^6, -3.9662761890979446\times 10^6, -60017.52423652596, 803515.1181825735, -2.091710365941658\times 10^6)$$
- **$\rho_2$** (Eq. B5):
  $$\lambda = (-40105.47653771657, 112253.0169706701, 23561.696065836168, -3.476180699403351\times 10^6, 1.137593670849482\times 10^7, 754313.1127166454, -1.308476044625268\times 10^7, 3.6444584853928134\times 10^7, 596226.612472288, -7.4277901143564405\times 10^6, 1.8928977514040343\times 10^7)$$
- **$\rho_3$** (Eq. B6):
  $$\lambda = (83208.35471266537, -191237.7264145924, -210916.2454782992, 8.71797508352568\times 10^6, -2.6914942420669552\times 10^7, -1.9889806527362722\times 10^6, 3.0888029960154563\times 10^7, -8.390870279256162\times 10^7, -1.4535031953446497\times 10^6, 1.7063528990822166\times 10^7, -4.2748659731120914\times 10^7)$$

### 6.2 Intermediate Amplitude Collocation Coefficient
- **$v_2$** (Collocation value at $f_2 = \frac{f_1 + f_3}{2}$, Eq. 21):
  $$\lambda = (0.8149838730507785, 2.5747553517454658, 1.1610198035496786, -2.3627771785551537, 6.771038707057573, 0.7570782938606834, -2.7256896890432474, 7.1140380397149965, 0.1766934149293479, -0.7978690983168183, 2.1162391502005153)$$

### 6.3 Merger-Ringdown Amplitude Coefficients
- **$\gamma_1$** (Eq. 19):
  $$\lambda = (0.006927402739328343, 0.03020474290328911, 0.006308024337706171, -0.12074130661131138, 0.26271598905781324, 0.0034151773647198794, -0.10779338611188374, 0.27098966966891747, 0.0007374185938559283, -0.02749621038376281, 0.0733150789135702)$$
- **$\gamma_2$** (Eq. 19):
  $$\lambda = (1.010344404799477, 0.0008993122007234548, 0.283949116804459, -4.049752962958005, 13.207828172665366, 0.10396278486805426, -7.025059158961947, 24.784892370130475, 0.03093202475605892, -2.6924023896851663, 9.609374464684983)$$
- **$\gamma_3$** (Eq. 19):
  $$\lambda = (1.3081615607036106, -0.005537729694807678, -0.06782917938621007, -0.6689834970767117, 3.403147966134083, -0.05296577374411866, -0.9923793203111362, 4.820681208409587, -0.006134139870393713, -0.38429253308696365, 1.7561754421985984)$$

### 6.4 Inspiral Phase Coefficients
- **$\sigma_1$** (Eq. 11):
  $$\lambda = (2096.551999295543, 1463.7493168261553, 1312.5493286098522, 18307.330017082117, -43534.1440746107, -833.2889543511114, 32047.31997183187, -108609.45037520859, 452.25136398112204, 8353.439546391714, -44531.3250037322)$$
- **$\sigma_2$** (Eq. 11):
  $$\lambda = (-10114.056472621156, -44631.01109458185, -6541.308761668722, -266959.23419307504, 686328.3229317984, 3405.6372187679685, -437507.7208209015, 1.6318171307344697\times 10^6, -7462.648563007646, -114585.25177153319, 674402.4689098676)$$
- **$\sigma_3$** (Eq. 11):
  $$\lambda = (22933.658273436497, 230960.00814979506, 14961.083974183695, 1.1940181342318142\times 10^6, -3.1042239693052764\times 10^6, -3038.166617199259, 1.8720322849093592\times 10^6, -7.309145012085539\times 10^6, 42738.22871475411, 467502.018616601, -3.064853498512499\times 10^6)$$
- **$\sigma_4$** (Eq. 11):
  $$\lambda = (-14621.71522218357, -377812.8579387104, -9608.682631509726, -1.7108925257214056\times 10^6, 4.332924601416521\times 10^6, -22366.683262266528, -2.5019716386377467\times 10^6, 1.0274495902259542\times 10^7, -85360.30079034246, -570025.3441737515, 4.396844346849777\times 10^6)$$

### 6.5 Intermediate Phase Coefficients
- **$\beta_1$** (Eq. 16):
  $$\lambda = (97.89747327985583, -42.659730877489224, 153.48421037904913, -1417.0620760768954, 2752.8614143665027, 138.7406469558649, -1433.6585075135881, 2857.7418952430758, 41.025109467376126, -423.680737974639, 850.3594335657173)$$
- **$\beta_2$** (Eq. 16):
  $$\lambda = (-3.282701958759534, -9.051384468245866, -12.415449742258042, 55.4716447709787, -106.05109938966335, -11.953044553690658, 76.80704618365418, -155.33172948098394, -3.4129261592393263, 25.572377569952536, -54.408036707740465)$$
- **$\beta_3$** (Eq. 16):
  $$\lambda = (-0.000025156429818799565, 0.000019750256942201327, -0.000018370671469295915, 0.000021886317041311973, 0.00008250240316860033, 7.157371250566708\times 10^{-6}, -0.000055780000112270685, 0.00019142082884072178, 5.447166261464217\times 10^{-6}, -0.00003220610095021982, 0.00007974016714984341)$$

### 6.6 Merger-Ringdown Phase Coefficients
- **$\alpha_1$** (Eq. 14):
  $$\lambda = (43.31514709695348, 638.6332679188081, -32.85768747216059, 2415.8938269370315, -5766.875169379177, -61.85459307173841, 2953.967762459948, -8986.29057591497, -21.571435779762044, 981.2158224673428, -3239.5664895930286)$$
- **$\alpha_2$** (Eq. 14):
  $$\lambda = (-0.07020209449091723, -0.16269798450687084, -0.1872514685185499, 1.138313650449945, -2.8334196304430046, -0.17137955686840617, 1.7197549338119527, -4.539717148261272, -0.049983437357548705, 0.6062072055948309, -1.682769616644546)$$
- **$\alpha_3$** (Eq. 14):
  $$\lambda = (9.5988072383479, -397.05438595557433, 16.202126189517813, -1574.8286986717037, 3600.3410843831093, 27.092429659075467, -1786.482357315139, 5152.919378666511, 11.175710130033895, -577.7999423177481, 1808.730762932043)$$
- **$\alpha_4$** (Eq. 14):
  $$\lambda = (-0.02989487384493607, 1.4022106448583738, -0.07356049468633846, 0.8337006542278661, 0.2240008282397391, -0.055202870001177226, 0.5667186343606578, 0.7186931973380503, -0.015507437354325743, 0.15750322779277187, 0.21076815715176228)$$
- **$\alpha_5$** (Eq. 14):
  $$\lambda = (0.9974408278363099, -0.007884449714907203, -0.059046901195591035, 1.3958712396764088, -4.516631601676276, -0.05585343136869692, 1.7516580039343603, -5.990208965347804, -0.017945336522161195, 0.5965097794825992, -2.0608879367971804)$$

---

## 7. Operational Boundaries & Thresholds

1. **Amplitude Boundaries:**
   - Inspiral to Intermediate: $Mf = 0.014$
   - Intermediate to Merger-Ringdown: $Mf = Mf_{\rm peak}$
   - Intermediate Collocation Points:
     $$f_1 = 0.014, \quad f_3 = Mf_{\rm peak}, \quad f_2 = \frac{f_1 + f_3}{2}$$

2. **Phase Boundaries:**
   - Inspiral to Intermediate: $Mf = 0.018$
   - Intermediate to Merger-Ringdown: $Mf = 0.5 f_{\rm RD}$

3. **High-Frequency Waveform Cutoff:**
   - $Mf_{\rm cut} = 0.2$

4. **Validity Boundaries:**
   - Final spin safety floor: $a_f \ge -0.717$
   - Mass ratio upper threshold: $q \le 5000$
