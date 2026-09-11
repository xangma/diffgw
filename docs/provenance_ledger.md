# TorchWave Literature Provenance and Coefficient Ledger

This document establishes the mathematical and literature provenance for all waveform approximants, calibration datasets, and surrogate models in TorchWave.

## 1. Clean-Room Architecture Principles

1. **Strict Literature Derivation**: All waveform formulas, Post-Newtonian coefficients, phenomenological ansatzes, and calibration parameters are derived exclusively from published peer-reviewed journals, public preprint archives (arXiv), and public journal supplemental material deposits.
2. **Zero Code Inspection or Translation**: No C, Python, or CUDA source code from LAL, LALSimulation, PyCBC, or LVK internal repositories was inspected, copied, translated, or adapted.
3. **Black-Box Binary Oracle Testing**: LALSimulation Python bindings are used solely as an external, black-box oracle to execute acceptance test gates against pre-computed physical parameter combinations.
4. **Transparent Residual Accounting**: Where clean-room implementations have unresolved physical discrepancies or missing table integrations, these are recorded openly as unmet gates rather than masked or artificially tuned.

---

## 2. Approximants and Primary Literature Provenance

### 2.1 TaylorF2 (Inspiral Aligned-Spin with Tides)
- **Primary References**:
  - Buonanno, Iyer, Ochsner, Pan, Sathyaprakash (2009), *Comparison of post-Newtonian templates for tests of general relativity*, Phys. Rev. D 80, 084043 ([DOI:10.1103/PhysRevD.80.084043](https://doi.org/10.1103/PhysRevD.80.084043))
  - Mishra, Kaviraj, Arun (2016), *Higher order spin-orbit and spin-spin post-Newtonian corrections to gravitational wave phasing*, Phys. Rev. D 93, 084039 ([DOI:10.1103/PhysRevD.93.084039](https://doi.org/10.1103/PhysRevD.93.084039))
  - Vines, Flanagan, Hinderer (2011), *Post-1-Newtonian tidal effects on gravitational waveforms from inspiralling compact binaries*, Phys. Rev. D 83, 084051 ([DOI:10.1103/PhysRevD.83.084051](https://doi.org/10.1103/PhysRevD.83.084051))
  - Damour, Nagar, Villain (2012), *Measurability of the tidal deformability of neutron stars with advanced detector networks*, Phys. Rev. D 85, 124037 ([DOI:10.1103/PhysRevD.85.124037](https://doi.org/10.1103/PhysRevD.85.124037))
- **Mathematical Form**:
  $$\tilde{h}(f) = \mathcal{A}(f) e^{-i \Psi(f)}$$
  $$\Psi(f) = 2\pi f t_c - \phi_c - \frac{\pi}{4} + \frac{3}{128\eta v^5} \sum_{k=0}^{7} \left(\psi_k + \psi_{k,\ln} \ln v\right) v^k + \Psi_{\rm tidal}(f)$$
  where $v = (\pi M f)^{1/3}$.
- **Phasing Alignment**:
  At reference frequency $f_{\rm ref}$, under LAL SPA convention:
  $$\Psi_{\rm LAL}(f_{\rm ref}) = \frac{3\pi}{4}$$
- **Verification Result**:
  Direct CPU FP64 complex relative $L_2$ error: $\mathbf{1.6 \times 10^{-12}}$ on both polarizations, with relative norm error $< 10^{-15}$ and flat mismatch $< 10^{-16}$.

### 2.2 TaylorF2_Eccentric (Eccentric Inspiral)
- **Primary References**:
  - Moore, Favata, Arun, Mishra (2016), *Gravitational-wave phasing for eccentric compact binaries in the post-Newtonian approximation*, Phys. Rev. D 93, 124061 ([DOI:10.1103/PhysRevD.93.124061](https://doi.org/10.1103/PhysRevD.93.124061))
  - Huerta et al. (2014), *Complete waveform model for eccentric compact binaries*, Phys. Rev. D 90, 084016 ([DOI:10.1103/PhysRevD.90.084016](https://doi.org/10.1103/PhysRevD.90.084016))
- **Mathematical Form**:
  $$\Psi_{\rm ecc}(f) = \Psi_{\rm circ}(f) - \frac{3}{128\eta v^5} e_0^2 \left(\frac{v_0}{v}\right)^{19/3} \sum_{j=0}^{6} \Delta\psi_j v^j$$
  Equations (4.17)–(4.18) of Moore et al. (2016).
- **Reference Frequency Priority**:
  $f_{0,\rm ecc}$ resolves hierarchically from: explicit $f_{0,\rm ecc} \to f_{\rm lower} \to f_{\rm ref} \to 20.0\text{ Hz}$, ensuring complete independence from numerical grid padding and DC sampling density.
- **Verification Result**:
  Direct CPU FP64 complex relative $L_2$ error: $\mathbf{1.6 - 2.0 \times 10^{-12}}$ on both polarizations.

### 2.3 IMRPhenomD (Aligned-Spin IMR)
- **Primary References**:
  - Husa, Khan, Hannam, Pürrer, Ohme, Forteza, Bohé (2016), *Frequency-domain gravitational waves from nonprecessing black-hole binaries. I. New numerical-relativity-informed amplitude and phase ansatzes*, Phys. Rev. D 93, 044006 ([DOI:10.1103/PhysRevD.93.044006](https://doi.org/10.1103/PhysRevD.93.044006)) [Paper I]
  - Khan, Husa, Hannam, Ohme, Pürrer, Forteza, Bohé (2016), *Frequency-domain gravitational waves from nonprecessing black-hole binaries. II. Phenomenological model for two-spin black holes*, Phys. Rev. D 93, 044007 ([DOI:10.1103/PhysRevD.93.044007](https://doi.org/10.1103/PhysRevD.93.044007)) [Paper II]
  - Berti, Cardoso, Will (2006), *Gravitational-wave spectroscopy of massive black holes with the space interferometer LISA: Quasinormal modes of spinning black holes*, Phys. Rev. D 73, 064030 ([DOI:10.1103/PhysRevD.73.064030](https://doi.org/10.1103/PhysRevD.73.064030))
- **Coefficients & Calibration**:
  - Table V of Khan et al. (2016) Paper II defines all 17 phenomenological parameters $\Lambda^i = \{\rho_1, \rho_2, \rho_3, v_2, \gamma_1, \gamma_2, \gamma_3, \sigma_1, \sigma_2, \sigma_3, \sigma_4, \beta_2, \beta_3, \alpha_2, \alpha_3, \alpha_4, \alpha_5\}$ as polynomials in $(\eta, \chi_{\rm PN} - 1)$ where $\chi_{\rm PN} = \chi_{\rm eff} - \frac{38\eta}{113}(\chi_1 + \chi_2)$.
  - Final remnant mass $M_f$ and spin $a_f$ fits from Husa et al. (2016) Eqs. (16)–(17).
  - QNM frequency $f_{\rm RD}$ and damping width $f_{\rm damp}$ from Berti et al. (2006) Table VIII ($l=2, m=2, n=0$).
- **Residual Analysis**:
  - Moderate-spin binaries match LALSimulation with optimized match $\ge 0.999999$.
  - In the high-spin corner $(30, 30, \chi_1=0.8, \chi_2=0.8)$, amplitude corrections reduce the magnitude relative $L_2$ error to $0.004309$ and bring the norm ratio to $1.000150$, with optimized match $0.9999744$ across $[20, 1024)\text{ Hz}, \Delta f=0.25\text{ Hz}, D=410\text{ Mpc}$. The residual is documented as an unmet gate for magnitude $L_2 \le 10^{-6}$.

### 2.4 IMRPhenomP / IMRPhenomPv2 (Precessing Waveforms)
- **Primary References**:
  - Hannam et al. (2014), *Twisting approximation for gravitational waves from precessing black-hole binaries*, Phys. Rev. Lett. 113, 151101 ([DOI:10.1103/PhysRevLett.113.151101](https://doi.org/10.1103/PhysRevLett.113.151101))
  - Schmidt, Hannam, Husa (2012), *Towards models of gravitational waveforms from generic binaries. II. Modelling precession through a single effective precession parameter*, Phys. Rev. D 86, 104063 ([DOI:10.1103/PhysRevD.86.104063](https://doi.org/10.1103/PhysRevD.86.104063))
  - Pratten et al. (2021), *Computationally efficient models for the dominant and subdominant harmonic modes of precessing binary black holes*, Phys. Rev. D 103, 104056 ([DOI:10.1103/PhysRevD.103.104056](https://doi.org/10.1103/PhysRevD.103.104056)) [Appendix B, Eqs. (B1)–(B2)]
- **Verification Result**:
  Optimized match $\ge 0.9982$ across precessing test configurations with inclination up to 0.3 rad.

### 2.5 IMRPhenomHM (Multipolar Higher Harmonics)
- **Primary References**:
  - London et al. (2018), *First higher-multipole model of gravitational waves from spinning and coalescing black-hole binaries*, Phys. Rev. Lett. 120, 161102 ([DOI:10.1103/PhysRevLett.120.161102](https://doi.org/10.1103/PhysRevLett.120.161102))
  - García-Quirós et al. (2020), *Multimode frequency-domain model for the gravitational-wave signal from nonprecessing black-hole binaries*, Phys. Rev. D 102, 064002 ([DOI:10.1103/PhysRevD.102.064002](https://doi.org/10.1103/PhysRevD.102.064002)) [Eq. (3)]
- **Common Time Shift Invariance**:
  Higher modes satisfy exact linear time translation:
  $$\tilde{h}_{\ell m}(f; t_c) = \tilde{h}_{\ell m}(f; 0) \exp(-2\pi i f t_c)$$
  Verified to relative error $< 2 \times 10^{-14}$ for all modes $(2,2), (2,1), (3,3), (3,2), (4,4)$.

### 2.6 IMRPhenomXAS & IMRPhenomXHM (Public Data Deposits)
- **Primary References**:
  - Pratten et al. (2020), *Setting the cornerstone for advanced gravitational-wave modeling: Accelerating IMRPhenomXAS*, Phys. Rev. D 102, 064001 ([DOI:10.1103/PhysRevD.102.064001](https://doi.org/10.1103/PhysRevD.102.064001))
  - García-Quirós et al. (2020), *Multimode frequency-domain model for the gravitational-wave signal from nonprecessing black-hole binaries*, Phys. Rev. D 102, 064002 ([DOI:10.1103/PhysRevD.102.064002](https://doi.org/10.1103/PhysRevD.102.064002))
- **Public Data Asset**:
  - The complete 26-fit parameter space Mathematica file was deposited with APS PRD:
    - **URL**: `https://journals.aps.org/prd/supplemental/10.1103/PhysRevD.102.064001/IMRPhenomXAS_Parameter_Space_Fits_Public_Release_Candidate.m`
    - **Path**: `src/torchwave/data/IMRPhenomXAS_Parameter_Space_Fits_Public_Release_Candidate.m`
    - **SHA256**: `94b9334c19f3a7909864891ea278ab9a411e4f33b7b7e17cc01d927056d9ac52`
    - **Size**: 61,053 bytes
    - **Access Status**: Freely downloadable public supplemental deposit.
    - **Integration Status**: Canonical version 104 fits and analytic inspiral amplitude boundary derivative are implemented in native PyTorch with parameter gradients and batch broadcasting. Five tested XAS configurations yielded component-optimized diagnostic matches 0.9960, 0.9988, 0.9986, 0.9990, and 0.9988, with magnitude relative $L_2$ errors $0.032 - 0.097$. This finite sample does not establish parameter-domain parity or continuity of the complete piecewise amplitude. See the [September 11 verification report](reviews/2026-09-11-agy/report.md) for current acceptance results.
  - **Phase 1 Clean-Room Parity Updates (2026-09-11)**:
    - *Analytic Amplitude Derivative*: Replaced the forward finite difference at the inspiral boundary $f_{\rm trans, amp}$ with analytic derivatives of the 3PN amplitude (Eq. 69) and pseudo-PN terms (Eq. 68). The PN helper was checked against independent complex-step differentiation to $< 10^{-11}$ relative error at five configurations; this does not independently verify the complete boundary construction.
    - *Intermediate Phase Collocation Weighted Average*: Checked published Eq. (95), $v_2^{\rm int} = 0.75(v_4^{\rm RD} + \Delta v_2^{\rm RD,4}) + 0.25 v_2^{\rm direct}$. The pre-dispatch formula was already correct; an attempted removal was rejected and restored during review.
    - *Parity Status*: Full acceptance gates ($10^{-6}$ relative $L_2$, $10^{-8}$ mismatch) remain unresolved; remaining amplitude discrepancies are not attributed to a single cause and require further investigation.
  - **Phase 2 Clean-Room Parity Acceptance Hardening (2026-09-11)**:
    - *Scoped Parity Verification*: Overhauled `tests/parity_harness.py` to enforce declarative frozen inventory validation, multi-detector (H1/L1/V1) projections using public LAL binary APIs (`lal.ComputeDetAMResponse`, `lal.TimeDelayFromEarthCenter`) at physical GPS times, multi-PSD coverage, and physical bounds checks.
    - *Catalog Scope*: Executed 16 deterministic cases covering 12 of the 17 manifest models: 6 analytic cases passed and 10 non-analytic cases failed. Five models without independently validated oracle mappings in this harness (`TaylorF2_HM`, `KerrRingdown`, `SEOBNR_ODE`, `SEOBNR_ROM`, `TEOBResumS`) are tracked as `UNRESOLVED_ORACLE`. Full catalog parity (`catalog_passed`) is `False`.
- **IMRPhenomXHM Multipolar Integration**:
  - `IMRPhenomXHM` uses the native `IMRPhenomXAS` carrier for the $(2, 2)$ base mode; `IMRPhenomHM` uses `IMRPhenomD`. The September 11 XHM catalog case has plus magnitude relative $L_2=0.11648$ and a component-optimized match of 0.99063. Strict parity remains unmet.

### 2.7 SEOBNR Reduced-Order Model (ROM Surrogate)
- **Primary References**:
  - Pürrer (2014), *Frequency domain reduced order models for aligned-spin effective-one-body waveforms*, Class. Quantum Grav. 31, 195010 ([DOI:10.1088/0264-9381/31/19/195010](https://doi.org/10.1088/0264-9381/31/19/195010))
  - Pürrer (2016), *Accelerating the evaluation of inspiral-merger-ringdown waveforms with reduced order modeling*, Phys. Rev. D 93, 064041 ([DOI:10.1103/PhysRevD.93.064041](https://doi.org/10.1103/PhysRevD.93.064041))
  - Bohé et al. (2017), *Improved effective-one-body model of spinning, nonprecessing binary black holes for the era of advanced detector observations with advanced LIGO*, Phys. Rev. D 95, 044028 ([DOI:10.1103/PhysRevD.95.044028](https://doi.org/10.1103/PhysRevD.95.044028))
- **Native Surrogate Asset**:
  - **Path**: `src/torchwave/data/seobnr_surrogate.pt`
  - **SHA256**: `8316be966b297f474be810024a6add2d5e64559ee3c506302e8ea83372085414`
  - **Generator**: `torchwave.approximants.eob.SEOBNR_ODE` (clean-room native PyTorch time-domain integrator)
  - **Frequency Coordinate**: Dimensionless geometric $M f = G M f / c^3 \in [0.0002, 0.30]$ (256 nodes)
  - **Architecture**: SVD basis projection with 12 amplitude basis vectors, 16 phase basis vectors, and degree-3 polynomial parameter mappings over $(\eta, \chi_1, \chi_2)$.
  - **Fidelity Statement**: The surrogate reproduces the native PyTorch SEOBNR_ODE generator. It is distinct from LAL's SEOBNRv4_ROM and does not use LAL training data.

### 2.8 NRTidal & NRTidalv2 (Tidal Deformability Models)
- **Primary References**:
  - Dietrich et al. (2017), *Gravitational-wave model for compact binaries with arbitrarily spinning neutron stars*, Phys. Rev. D 96, 121501 ([DOI:10.1103/PhysRevD.96.121501](https://doi.org/10.1103/PhysRevD.96.121501))
  - Dietrich et al. (2019), *Matter matters: A improved analytical model for binary neutron star mergers*, Phys. Rev. D 100, 044003 ([DOI:10.1103/PhysRevD.100.044003](https://doi.org/10.1103/PhysRevD.100.044003)) [Eqs. (15)–(21)]

### 2.9 IMRPhenomNSBH (Neutron Star-Black Hole Mergers)
- **Primary References**:
  - Pannarale et al. (2015), *Gravitational-wave model for black-hole-neutron-star mergers*, Phys. Rev. D 92, 084041 ([DOI:10.1103/PhysRevD.92.084041](https://doi.org/10.1103/PhysRevD.92.084041))
  - Thompson et al. (2020), *Phenomenological model for the gravitational-wave signal from black-hole-neutron-star mergers*, Phys. Rev. D 101, 124059 ([DOI:10.1103/PhysRevD.101.124059](https://doi.org/10.1103/PhysRevD.101.124059))

---

## 3. Machine-Readable Asset Checksum Ledger

| Asset | Path | SHA256 Checksum | Provenance / Origin |
|---|---|---|---|
| Native SEOBNR ROM Surrogate | `src/torchwave/data/seobnr_surrogate.pt` | `8316be966b297f474be810024a6add2d5e64559ee3c506302e8ea83372085414` | Trained via `torchwave.approximants.eob.SEOBNR_ODE` on CPU |
| APS IMRPhenomXAS Parameter Space Fits | `src/torchwave/data/IMRPhenomXAS_Parameter_Space_Fits_Public_Release_Candidate.m` | `94b9334c19f3a7909864891ea278ab9a411e4f33b7b7e17cc01d927056d9ac52` | APS PRD Supplemental Material (Pratten et al. 2020) |
