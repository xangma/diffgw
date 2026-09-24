# TorchWave Literature Provenance and Coefficient Ledger

This document records mathematical sources, asset provenance, and calibration
provenance in TorchWave. Model-specific comparisons with LALSimulation are
recorded on [Differences from LAL](differences_from_lal.md#model-specific-differences).
Numerical agreement, physical calibration and software/data provenance are separate
claims; evidence for one does not establish the others.

## 1. Source and Validation Policy

1. **Mathematical and software sources**: Model implementations draw on primary literature, public author data and explicitly attributed permissive software adaptations. Sources and unresolved inherited calibration provenance are recorded here and in [third-party notices](../THIRD_PARTY_NOTICES.md).
2. **Reference comparisons**: LALSimulation binaries provide external waveform comparisons. Historical implementation and reference-audit records are preserved on `archive/clean-room-reviews`; these records do not establish clean-room provenance for every inherited coefficient.
3. **Calibration and numerical conventions**: Native NR fits use the identified SXS data. Numerical compatibility conventions, such as QNM rounding and phase/time choices, are documented separately from physical calibration.
4. **Scope of agreement**: Passing a declared case establishes agreement for its parameters, modes, frequency grid, precision and reference version. It does not establish exact equality or accuracy throughout a model's parameter domain.

---

## 2. Approximants and Primary Literature Provenance

### Implementation and Evidence Summary

- **Reference comparisons**: The archived CPU FP64 catalog passed **17 of 21 cases**; the holdout suite passed **11 of 13**. Both suites report failure overall. The records include case definitions, reference versions, unaligned complex errors and detector projections: [catalog][frozen-catalog], [holdouts][frozen-holdout], [source record][frozen-source]. These are historical snapshots, not fresh results for the current checkout.
- **Native NR components**: Ringdown amplitude and phase fits cover $(2,1)$, $(3,3)$, $(4,4)$ and $(3,2)$. The phase tests below isolate the ringdown phase using reference amplitudes and restricted frequency bands; they do not validate complete multimode waveforms.
- **Continuous EOB attachment**: Natural cubic splines locate an interior orbital-frequency maximum on accepted integration samples. Removing an auxiliary resampling grid does not remove dependence on integration accuracy or spline resolution.
- **Shared physics and frontend checks**: `diffgw_core` contains framework-independent Python/NumPy constants and coefficients. Tests in `tests/parity/` compare the Torch and JAX implementations with per-test tolerances; frontend agreement alone is not an independent physical validation.

The native waveform implementations do not call LAL at runtime. The canonical
`SEOBNR_ROM` path requires external official ROM data. Original software is MIT
licensed; third-party data and adaptations retain their identified terms, and
some inherited calibration provenance remains unresolved. Differentiability and
batching depend on the selected model and numerical path.

The [retired prototype record](waveforms/research/retired-release-surface.md)
preserves the former complete-waveform routes and native pilot asset metadata.
Historical comparisons below remain evidence about those former implementations,
not validation of the retained official ROM or surviving equation components.

[frozen-catalog]: https://github.com/xangma/diffgw/blob/0be15a5b80dd5383da054165f6834461798d70dc/docs/reviews/2026-09-13-parity-review/verification/reference-contracts-checkpoint3/catalog.json
[frozen-holdout]: https://github.com/xangma/diffgw/blob/0be15a5b80dd5383da054165f6834461798d70dc/docs/reviews/2026-09-13-parity-review/verification/reference-contracts-checkpoint3/holdout.json
[frozen-source]: https://github.com/xangma/diffgw/blob/0be15a5b80dd5383da054165f6834461798d70dc/docs/reviews/2026-09-13-parity-review/verification/reference-contracts-checkpoint3/source-record.json

### 2.1 TaylorF2 (Inspiral Aligned-Spin with Tides)
- **Executable mathematical reference**: The [nonspinning 3.5PN specification](math/taylorf2.md), [aligned-spin black-hole extension](math/taylorf2-aligned-spin.md), [adiabatic quadrupolar tidal extension](math/taylorf2-tidal.md), and [spin-induced quadrupole extension](math/taylorf2-spin-quadrupole.md) connect equation-level sources, explicit conventions, independent SymPy/mpmath expressions, and Torch/JAX coefficient, waveform, and derivative checks. Each extension reuses unchanged baseline components and restricted amplitude. The last two add 5PN/6PN/6.5PN electric tidal and non-Kerr 2PN/3PN/3.5PN spin-quadrupole phase terms. Independent spin-induced octupoles, spin-tidal couplings, magnetic/dynamical tides, absorption, amplitude corrections, and termination remain outside these specifications.
- **Primary References**:
  - Buonanno, Iyer, Ochsner, Pan, Sathyaprakash (2009), *Comparison of post-Newtonian templates for compact binary inspiral signals in gravitational-wave detectors*, Phys. Rev. D 80, 084043 ([DOI:10.1103/PhysRevD.80.084043](https://doi.org/10.1103/PhysRevD.80.084043))
  - Mishra, Kela, Arun, Faye (2016), *Ready-to-use post-Newtonian gravitational waveforms for binary black holes with non-precessing spins: An update*, Phys. Rev. D 93, 084054 ([DOI:10.1103/PhysRevD.93.084054](https://doi.org/10.1103/PhysRevD.93.084054))
  - Wade et al. (2013), [arXiv:1306.3901v2](https://arxiv.org/abs/1306.3901v2), Eq. (4) and spin-parameter definitions; lower-order aligned-spin terms with black-hole quadrupoles.
  - Bohé et al. (2015), [arXiv:1501.01529v1](https://arxiv.org/abs/1501.01529v1), Eqs. (3.33), (4.14), and Messina–Nagar (2017), [arXiv:1703.08107v4](https://arxiv.org/abs/1703.08107v4), corrected Eq. (26); energy-balance derivation of the complete 3.5PN quadratic-spin phase term.
  - Pratten et al. (2020), [arXiv:2001.11412v1](https://arxiv.org/abs/2001.11412v1), Eqs. (A19), (A23), (A24); the baseline phase and its explicit spin-spin tail addition, independently checked through energy balance. The aligned-spin specification records the correction of the previously omitted (A24) term.
  - Vines, Hinderer, Flanagan (2011), *Post-1-Newtonian tidal effects in the gravitational waveform from binary inspirals*, Phys. Rev. D 83, 084051 ([DOI:10.1103/PhysRevD.83.084051](https://doi.org/10.1103/PhysRevD.83.084051))
  - Wade et al. (2014), [arXiv:1402.5156v1](https://arxiv.org/abs/1402.5156v1), Eqs. (4)–(6), (A26); equivalent effective-parameter and component-mass expressions for the 5PN/6PN tidal phase.
  - Damour, Nagar, Villain (2012), *Measurability of the tidal polarizability of neutron stars in late-inspiral gravitational-wave signals*, Phys. Rev. D 85, 123007 ([DOI:10.1103/PhysRevD.85.123007](https://doi.org/10.1103/PhysRevD.85.123007))
    The tidal reference pins [arXiv:1203.4352v1](https://arxiv.org/abs/1203.4352v1), Eqs. (B8)–(B12), (B14), including the 6.5PN tail and tidal-coupling conversion.
  - Nagar et al. (2019), [arXiv:1812.07923v1](https://arxiv.org/pdf/1812.07923v1), Eqs. (44), (46), (47), (50)–(52); spin-induced quadrupole corrections at 2PN, 3PN, and the 3.5PN tail.
  - Dietrich et al. (2019), [arXiv:1905.06011v2](https://arxiv.org/pdf/1905.06011v2), Eqs. (26)–(27); quadrupole-dependent cubic-spin correction. Marsat, [arXiv:1411.4118v2](https://arxiv.org/pdf/1411.4118v2), Eqs. (6.20)–(6.21), supplies an independent cubic-spin cross-check.
- **Mathematical Form**:
  $$\tilde{h}(f) = \mathcal{A}(f) e^{-i \Psi(f)}$$
  $$\Psi_{\rm coal}(f) = 2\pi f t_c - 2\phi_c - \frac{\pi}{4} + \frac{3}{128\eta v^5} \sum_{k=0}^{7} \left(\psi_k + \psi_{k,\ln} \ln v\right) v^k + \Psi_{\rm tidal}(f)$$
  where $v = (\pi M f)^{1/3}$ and `phic` is the orbital phase. This expression uses the coalescence convention.
- **Reference-Frequency Phasing**:
  The reference-frequency convention subtracts the intrinsic phase at $f_{\rm ref}$ and adds $3\pi/4$, so
  $$\Psi(f_{\rm ref}) = \frac{3\pi}{4} + 2\pi f_{\rm ref} t_c - 2\phi_c.$$
  It reduces to $3\pi/4$ only when $t_c=\phi_c=0$.
- **Verification Result**:
  The four [frozen catalog cases][frozen-catalog] passed the declared CPU FP64 gates. The largest polarization complex relative $L_2$ error was approximately $1.64 \times 10^{-12}$; detector-projection errors are reported separately in the artifact.

### 2.2 TaylorF2_Eccentric (Eccentric Inspiral)
- **Primary References**:
  - Moore, Favata, Arun, Mishra (2016), *Gravitational-wave phasing for low-eccentricity inspiralling compact binaries to 3PN order*, Phys. Rev. D 93, 124061 ([DOI:10.1103/PhysRevD.93.124061](https://doi.org/10.1103/PhysRevD.93.124061))
  - Huerta et al. (2014), *Accurate and efficient waveforms for compact binaries on eccentric orbits*, Phys. Rev. D 90, 084016 ([DOI:10.1103/PhysRevD.90.084016](https://doi.org/10.1103/PhysRevD.90.084016))
- **Mathematical Form**:
  $$\Psi_{\rm ecc}(f) = \Psi_{\rm circ}(f) - \frac{3}{128\eta v^5} e_0^2 \left(\frac{v_0}{v}\right)^{19/3} \sum_{j=0}^{6} \Delta\psi_j v^j$$
  Equation (6.26) of Moore et al. (2016) gives the eccentric SPA phase. The optional native amplitude factor is a project option whose primary derivation remains pending.
- **Reference Frequency Priority**:
  $f_{0,\rm ecc}$ resolves from an explicit value, then $f_{\rm lower}$, then a positive $f_{\rm ref}$, then the lowest positive frequency-grid sample (20 Hz if none exists). The final grid fallback makes the result dependent on the supplied positive grid.
- **Verification Result**:
  The two [frozen catalog cases][frozen-catalog] passed, with a maximum polarization complex relative $L_2$ error of approximately $1.64 \times 10^{-12}$.

### 2.3 IMRPhenomD (Aligned-Spin IMR)
- **Primary References**:
  - Husa, Khan, Hannam, Pürrer, Ohme, Forteza, Bohé (2016), *Frequency-domain gravitational waves from non-precessing black-hole binaries. I. New numerical waveforms and anatomy of the signal*, Phys. Rev. D 93, 044006 ([DOI:10.1103/PhysRevD.93.044006](https://doi.org/10.1103/PhysRevD.93.044006)) [Paper I]
  - Khan, Husa, Hannam, Ohme, Pürrer, Forteza, Bohé (2016), *Frequency-domain gravitational waves from non-precessing black-hole binaries. II. A phenomenological model for the advanced detector era*, Phys. Rev. D 93, 044007 ([DOI:10.1103/PhysRevD.93.044007](https://doi.org/10.1103/PhysRevD.93.044007)) [Paper II]
  - Berti, Cardoso, Will (2006), *On gravitational-wave spectroscopy of massive black holes with the space interferometer LISA*, Phys. Rev. D 73, 064030 ([DOI:10.1103/PhysRevD.73.064030](https://doi.org/10.1103/PhysRevD.73.064030))
- **Coefficients & Calibration**:
  - Table V of Khan et al. (2016) Paper II prints 19 phenomenological parameter rows $\Lambda^i$ as polynomials in $(\eta, \chi_{\rm PN} - 1)$ where $\chi_{\rm PN} = \chi_{\rm eff} - \frac{38\eta}{113}(\chi_1 + \chi_2)$. The waveform implementation uses 18 fitted rows and obtains $\beta_1$ from its phase-join condition.
  - The [first executable amplitude specification](math/imrphenomd-amplitude.md) uses the seven amplitude rows at the digits printed in Table V. The production table retains additional inherited digits whose precise publication provenance remains unresolved. Algebra checks using those digits are conditional and do not establish their source.
  - The [executable phase specification](math/imrphenomd-phase.md) transcribes the 12 printed phase rows, Appendix B aligned-spin PN terms, three phase branches and two continuity joins from Paper II. The backend checks explicitly supply inherited extra fit digits, so their algebraic agreement does not establish those digits' provenance. The frontend's amplitude-peak time gauge is recorded as a separate implementation convention.
  - The [composed polarization specification](math/imrphenomd-waveform.md) combines those unchanged slices with the independently audited SI/IAU conversions, the physical stationary-phase normalization of [Cutler and Flanagan, Eqs. (62)–(63)](https://arxiv.org/pdf/gr-qc/9402014v1), and explicit DiffGW phase/polarization conventions. Direct complex-strain and selected smooth-branch derivative checks against both frontends use a declared QNM pair and explicitly supplied inherited fit rows; they do not establish fit provenance or physical accuracy.
  - The [full intrinsic composition](math/imrphenomd-intrinsic.md) supplies the QNM pair from the independent remnant expressions and declared numerical spline data. At fixed total mass, it checks selected full-chain mass-ratio and aligned-spin complex-strain derivatives against both frontends, including the moving remnant, QNM inputs, joins, and phase gauge. This now includes both signed high-spin Kerr spline branches and an interior-spin point on the active remnant cap, with guarded smooth-region scalar stencils. It also compares complete high-spin complex waveforms with an independently evaluated generated Kerr spline. Its conditional inherited-coefficient comparisons do not recover publication provenance for the extra digits or establish physical accuracy.
  - The [executable remnant and QNM-input specification](math/imrphenomd-remnant-qnm.md) transcribes Husa et al. arXiv:1508.07250v2 Eqs. (22)–(24) for final spin, radiated energy and mass, and Eq. (17) for QNM mass scaling. The implementation retains additional inherited fit digits; one spin coefficient does not round to the paper's printed digit.
  - An [independent four-term final-spin refit](math/imrphenomd-remnant-qnm.md#independent-final-spin-recalibration) uses the later, author-published 427-case NR table with pinned input and script hashes. Weight and fitting choices move the disputed coefficient substantially; this experiment does not reconstruct the original 106-case calibration or change the production coefficient.
  - The current QNM frequency and damping representation uses independently generated Kerr eigenvalues, rounded to six significant digits and interpolated with a natural cubic spline in `kerr_qnm_d_sampling_clean.py`; outside its compatibility interval, it uses the generated eigenvalue evaluator. Berti et al. (2006) provides background and the optional legacy fit representation.
- **Verification Result**:
  The declared [catalog][frozen-catalog] and [holdout][frozen-holdout] cases passed the complex relative $L_2 \le 10^{-6}$ and mismatch $\le 10^{-8}$ gates. This result applies to those configurations.

### 2.4 IMRPhenomP / IMRPhenomPv2 (Precessing Waveforms)
- **Primary References**:
  - Hannam et al. (2014), *A simple model of complete precessing black-hole-binary gravitational waveforms*, Phys. Rev. Lett. 113, 151101 ([DOI:10.1103/PhysRevLett.113.151101](https://doi.org/10.1103/PhysRevLett.113.151101))
  - Schmidt, Hannam, Husa (2012), *Towards models of gravitational waveforms from generic binaries: A simple approximate mapping between precessing and non-precessing inspiral signals*, Phys. Rev. D 86, 104063 ([DOI:10.1103/PhysRevD.86.104063](https://doi.org/10.1103/PhysRevD.86.104063))
  - Pratten et al. (2021), *Computationally efficient models for the dominant and sub-dominant harmonic modes of precessing binary black holes*, Phys. Rev. D 103, 104056 ([DOI:10.1103/PhysRevD.103.104056](https://doi.org/10.1103/PhysRevD.103.104056)) [Appendix B, Eqs. (B1)–(B2)]
- **Verification Result**:
  The `IMRPhenomPv2` [catalog][frozen-catalog] and [holdout][frozen-holdout] cases passed the complex relative $L_2 \le 10^{-6}$ and mismatch $\le 10^{-8}$ gates. This does not establish a separate result for every PhenomP variant.

### 2.4a IMRPhenomXP MSA v223 convention-6 output

- **Published convention:** [Pratten et al. (2021), Appendix C Eqs. (67), (79), (81) and Appendix F Table IV](https://arxiv.org/html/2004.06503v2) specify the dominant-mode twist-up, source-frame rotation, polarization basis and convention-dependent Euler offsets and carrier phase. Table IV assigns zero carrier phase to convention 1 and `phic` to convention 6.
- **Declared native mapping:** For the quadrupole-only XP v223, `final_spin_mod=3`, `phi0=0` route, the complete convention-6 polarizations equal the clean convention-1 polarizations times $\exp(2i\,\mathrm{phic})$. This is an output convention identity after the full rotation and basis projection, not a change to the MSA dynamics or a higher-mode rule. The clean helper remains defined only for conventions 0 and 1.
- **Compatibility and limits:** The generic dense-grid convention-6 fixture remains below $10^{-8}$ complex relative $L_2$ against the public LAL v223 binary. The aligned fixture separates two contracts: the explicit v223 helper retains its strict binary gate, while the physical public opening has a measured $10^{-5}$–$5\times10^{-5}$ residual. The v223 auxiliary-root endpoint remains only in that compatibility diagnostic; the `L+S_parallel <= 0` guard remains in public dispatch. Generic two-spin nonzero `phi0`, other precession versions and other remnant choices retain the established XP route. The frozen catalog and holdout qualification numbers below cover convention 1; these additional probes do not certify a parameter domain or JAX convention-6 dispatch. See the [physical boundary analysis](waveforms/research/xp-v223-reference-closure.md#physical-zero-nutation-continuation-2026-09-24).

### 2.5 IMRPhenomHM (Multipolar Higher Harmonics)
- **Primary References**:
  - London et al. (2018), *First higher-multipole model of gravitational waves from spinning and coalescing black-hole binaries*, Phys. Rev. Lett. 120, 161102 ([DOI:10.1103/PhysRevLett.120.161102](https://doi.org/10.1103/PhysRevLett.120.161102))
  - García-Quirós et al. (2020), *IMRPhenomXHM: A multi-mode frequency-domain model for the gravitational wave signal from non-precessing black-hole binaries*, Phys. Rev. D 102, 064002 ([DOI:10.1103/PhysRevD.102.064002](https://doi.org/10.1103/PhysRevD.102.064002)) [Eq. (3)]
- **Common Time Shift Invariance**:
  Higher modes satisfy exact linear time translation:
  $$\tilde{h}_{\ell m}(f; t_c) = \tilde{h}_{\ell m}(f; 0) \exp(-2\pi i f t_c)$$
  The declared [catalog][frozen-catalog] and [holdout][frozen-holdout] comparisons passed the $L_2 \le 10^{-6}$ and mismatch $\le 10^{-8}$ gates; these finite-tolerance comparisons do not establish exact equality.

### 2.6 IMRPhenomXAS & IMRPhenomXHM (Public Data Deposits & Native Calibration)
- **Primary References**:
  - Pratten et al. (2020), *Setting the cornerstone for the IMRPhenomX family of models for gravitational waves from compact binaries: The dominant harmonic for non-precessing quasi-circular black holes*, Phys. Rev. D 102, 064001 ([DOI:10.1103/PhysRevD.102.064001](https://doi.org/10.1103/PhysRevD.102.064001))
  - García-Quirós et al. (2020), *IMRPhenomXHM: A multi-mode frequency-domain model for the gravitational wave signal from non-precessing black-hole binaries*, Phys. Rev. D 102, 064002 ([DOI:10.1103/PhysRevD.102.064002](https://doi.org/10.1103/PhysRevD.102.064002))
- **Public Data Asset for IMRPhenomXAS**:
  - The complete 26-fit parameter space Mathematica file deposited with APS PRD:
    - **URL**: `https://journals.aps.org/prd/supplemental/10.1103/PhysRevD.102.064001/IMRPhenomXAS_Parameter_Space_Fits_Public_Release_Candidate.m`
    - **SHA256**: `94b9334c19f3a7909864891ea278ab9a411e4f33b7b7e17cc01d927056d9ac52`
    - **Access Status**: Freely downloadable public supplemental deposit.
    - **Executable fit check**: The opt-in [XAS author-fit verifier](https://github.com/xangma/diffgw/blob/main/scripts/verify_xas_author_fits.py) checks the pinned file without redistributing it. Across nine mass-ratio/spin cases, 24 parameter surfaces and the two `(2,2)` QNM surfaces agree in value with both native frontends; selected parameter and QNM spin derivatives agree with independent complex-step derivatives. Two of these parameter surfaces are available for the fifth-order amplitude alternative but are not called by the quartic carrier. The intermediate-amplitude midpoint fit is an explicit carrier variant: the deposit uses `chiPNhat` and the native carrier uses `sTotR`, which the verifier checks separately. This establishes source-fit transcription on sampled inputs, not independent NR calibration or physical accuracy; see [the mathematical specification](math/imrphenomxas.md#author-fit-surface-cross-check).
  - **Reference Result**: The single [frozen catalog case][frozen-catalog] had maximum polarization complex relative $L_2$ error approximately $4.21 \times 10^{-14}$. The two [holdout cases][frozen-holdout] reached approximately $9.49 \times 10^{-9}$; these figures describe the recorded configurations, not a uniform accuracy bound.
- **IMRPhenomXHM Native NR Calibration**:
  - Fits ringdown surfaces to SXS numerical-relativity data across $(\ell, m) \in \{(2,1), (3,3), (4,4), (3,2)\}$, providing smooth amplitude profiles and continuous phase matching.
  - Uses native Kerr Leaver quasinormal mode frequencies and damping rates from `torchwave.qnm`.
  - Phase-component test gates and their restricted scope are described in Section 2.10. The historical full-waveform LAL comparisons did not pass the strict gates.

### 2.7 SEOBNR Reduced-Order Model (ROM Surrogate)
- **Primary References**:
  - Pürrer (2014), *Frequency domain reduced order models for gravitational waves from aligned-spin compact binaries*, Class. Quantum Grav. 31, 195010 ([DOI:10.1088/0264-9381/31/19/195010](https://doi.org/10.1088/0264-9381/31/19/195010))
  - Pürrer (2016), *Frequency domain reduced order model of aligned-spin effective-one-body waveforms with generic mass-ratios and spins*, Phys. Rev. D 93, 064041 ([DOI:10.1103/PhysRevD.93.064041](https://doi.org/10.1103/PhysRevD.93.064041))
  - Bohé et al. (2017), *An improved effective-one-body model of spinning, nonprecessing binary black holes for the era of gravitational-wave astrophysics with advanced detectors*, Phys. Rev. D 95, 044028 ([DOI:10.1103/PhysRevD.95.044028](https://doi.org/10.1103/PhysRevD.95.044028))
- **Native development**: The unvalidated native surrogate and SEOBNR waveform
  assemblers have been removed from the release package. They are retained on
  `codex/native-seobnr-rom` for generator qualification before any new tables
  are trained or promoted. The former `seobnrv5rom` alias is no longer accepted.
- **Official ROM Wrapper**:
  - The canonical `SEOBNR_ROM` / `SEOBNRv4ROM` aliases use tensor reconstruction from the external official `SEOBNRv4ROM_v3.0.hdf5` data. The data are required for this path and are not bundled.
  - Registered in `torchwave.external_data` from Zenodo record 14999310 (SHA-256 `4ff5317967d757fcad9323a74b75f4f63894ed87f4b1852d46cbe47cde488b9c`); redistribution terms remain unresolved.
  - The single [frozen catalog case][frozen-catalog] had maximum polarization complex relative $L_2$ error approximately $1.20 \times 10^{-11}$.

### 2.8 NRTidal & NRTidalv2 (Tidal Deformability Models)
- **Primary References**:
  - Dietrich et al. (2017), *Closed-form tidal approximants for binary neutron star gravitational waveforms constructed from high-resolution numerical relativity simulations*, Phys. Rev. D 96, 121501 ([DOI:10.1103/PhysRevD.96.121501](https://doi.org/10.1103/PhysRevD.96.121501))
  - Dietrich et al. (2019), *Improving the NRTidal model for binary neutron star systems*, Phys. Rev. D 100, 044003 ([DOI:10.1103/PhysRevD.100.044003](https://doi.org/10.1103/PhysRevD.100.044003)) [Eqs. (15)–(21)]
- **Reference Result**: The declared [catalog][frozen-catalog] and [holdout][frozen-holdout] cases for `IMRPhenomD_NRTidal` and `IMRPhenomD_NRTidalv2` passed $L_2 \le 10^{-6}$ and mismatch $\le 10^{-8}$ gates.

### 2.9 IMRPhenomNSBH (Neutron Star-Black Hole Mergers)
- **Primary References**:
  - Pannarale et al. (2015), *Aligned spin neutron star-black hole mergers: a gravitational waveform amplitude model*, Phys. Rev. D 92, 084050 ([DOI:10.1103/PhysRevD.92.084050](https://doi.org/10.1103/PhysRevD.92.084050))
  - Thompson et al. (2020), *Modeling the gravitational wave signature of neutron star black hole coalescences: PhenomNSBH*, Phys. Rev. D 101, 124059 ([DOI:10.1103/PhysRevD.101.124059](https://doi.org/10.1103/PhysRevD.101.124059))
- **Reference Result**: The declared [catalog][frozen-catalog] and [holdout][frozen-holdout] cases passed $L_2 \le 10^{-6}$ and mismatch $\le 10^{-8}$ gates against `IMRPhenomNSBH`.

### 2.10 Native SXS NR Higher-Harmonic Calibrations (IMRPhenomXHM & XPHM)
- **Primary References**:
  - García-Quirós et al. (2020), *IMRPhenomXHM: A multi-mode frequency-domain model for the gravitational wave signal from non-precessing black-hole binaries*, Phys. Rev. D 102, 064002 ([DOI:10.1103/PhysRevD.102.064002](https://doi.org/10.1103/PhysRevD.102.064002))
  - SXS Collaboration Catalog (Simulating eXtreme Spacetimes, CC-BY-4.0)
- **Ringdown Lorentzian Amplitude Fits**:
  $$A_{\ell m}(f) = f^{-1/12} \frac{|A_\lambda| \sigma f_{\rm damp}}{(f - f_{\rm ring})^2 + (\sigma f_{\rm damp})^2} \exp\left[-\frac{\lambda}{\sigma}\frac{f - f_{\rm ring}}{f_{\rm damp}}\right]$$
  - $(2, 1)$: $A_\lambda = 0.057335 \eta \delta$, $\lambda = 0.887167$, $\sigma = 1.0$
  - $(3, 3)$: $A_\lambda = 0.203728 \eta \delta$, $\lambda = 0.771190$, $\sigma = 1.0$
  - $(4, 4)$: $A_\lambda = 0.098380 \eta^2$, $\lambda = 0.938766$, $\sigma = 1.0$
  - $(3, 2)$: $A_\lambda = 0.052026 \eta^2$, $\lambda = 0.846467$, $\sigma = 1.0$
- **Ringdown Phase Fits in $x = 4\eta - 1$**:
  $$\phi_{\ell m}(f) = -\alpha_2 \frac{f_{\rm ring}^2}{f} + \alpha_\lambda \arctan\left(\frac{f - f_{\rm ring}}{f_{\rm damp}}\right) + \Delta t_{\rm rd} f + \phi_0$$
  - $(2, 1)$: $\alpha_2 = -30.0000 x + 10.0000$, $\alpha_\lambda = -4.6194 x - 4.3817$, $t_0 = -4.5672 M$
  - $(3, 3)$: $\alpha_2 = -7.9241 x + 37.1501$, $\alpha_\lambda = -0.0798 x - 1.1297$, $t_0 = -13.7830 M$
  - $(4, 4)$: $\alpha_2 = 3.6206 x + 63.1662$, $\alpha_\lambda = 5.8946 x - 0.4360$, $t_0 = -16.1609 M$
  - $(3, 2)$: $\alpha_2 = 5.8014 x + 39.0886$, $\alpha_\lambda = -1.4100 x - 1.8403$, $t_0 = -16.8168 M$
- **Verification Gates**:
  [`tests/test_xhm_native_fits.py`](https://github.com/xangma/diffgw/blob/main/tests/test_xhm_native_fits.py) defines phase-component checks using the **reference amplitude**, restricted ringdown frequency bands and affine phase/time alignment:
  - Seven $(3,3)$ holdouts (`SXS:BBH:2265`, `2483`, `2484`, `2486`, `2489`, `2491`, `0185`, with $3 \le q \le 10$): phase RMS $< 0.020$ rad and fitted time-shift magnitude $< 0.20 M$; tighter phase gates apply at $q \le 3.5$.
  - Two additional holdouts (`SXS:BBH:0030`, `0166`, with $q=3,6$) for the four modes: phase RMS $< 0.15$ rad and affine mismatch $< 0.010$. These tests do not impose the $(3,3)$ arrival-time gate on every mode.
  These are test thresholds, not a new measurement. The tests skip when their external data or archived manifest are unavailable. Complete waveform amplitude, inspiral and precession accuracy are outside these phase-only checks.

### 2.11 Continuous NQC Spline Attachment (EOB components)
- **Primary References**:
  - Bohé et al. (2017), *An improved effective-one-body model of spinning, nonprecessing binary black holes for the era of gravitational-wave astrophysics with advanced detectors*, Phys. Rev. D 95, 044028 ([DOI:10.1103/PhysRevD.95.044028](https://doi.org/10.1103/PhysRevD.95.044028))
  - Nagar et al. (2018), *Time-domain effective-one-body gravitational waveforms for coalescing compact binaries with nonprecessing spins, tides and self-spin effects*, Phys. Rev. D 98, 104052 ([DOI:10.1103/PhysRevD.98.104052](https://doi.org/10.1103/PhysRevD.98.104052))
- **Mathematical Form**:
  `src/torchwave/approximants/eob/continuous_matching.py` locates a unique interior maximum of a natural cubic spline through accepted orbital-frequency samples by solving $d\Omega/dt = 0$. The selected root is locally differentiable; interval selection is piecewise and unresolved or tied maxima are rejected. The attachment tests check continuity and independence from an auxiliary resampling spacing with the accepted samples fixed. Integration-step convergence and complete EOB waveform accuracy require separate checks; the frozen EOB LAL cases did not pass.

### 2.12 Clean-Room Kerr Quasinormal Mode Solvers & Generation (`torchwave.qnm`)
- **Primary References**:
  - Leaver (1985), *An analytic representation for the quasinormal modes of Kerr black holes*, Proc. R. Soc. Lond. A 402, 285–298 ([DOI:10.1098/rspa.1985.0119](https://doi.org/10.1098/rspa.1985.0119))
  - Cook & Zalutskiy (2014), *Gravitational perturbations of the Kerr geometry: High-accuracy study*, Phys. Rev. D 90, 124021 ([arXiv:1410.7698](https://arxiv.org/abs/1410.7698))
  - London et al. (2018), *First higher-multipole model of gravitational waves from spinning and coalescing black-hole binaries*, Phys. Rev. Lett. 120, 161102 ([DOI:10.1103/PhysRevLett.120.161102](https://doi.org/10.1103/PhysRevLett.120.161102))
- **Mathematical Form**:
  - **Radial Equation**: Spin $-2$ Kerr radial Teukolsky continued fractions solved via Cook & Zalutskiy Eqs. (23)–(44) with high-order asymptotic remainder conditioning.
  - **Angular Eigenvalues**: Computed via spectral matrix diagonalization of Cook & Zalutskiy Eqs. (49)–(56), supporting exact spherical branches and continuous branch continuation.
  - **Full Kerr Table Generator** (`torchwave.qnm.generate` / `torchwave-generate-qnm`): Generates all 16 positive-$m$ and negative-$m$ low overtones up to $n=2$ across $a \in [0, 0.99999]$ using adaptive degree-5 interpolating splines (`src/torchwave/approximants/kerr_qnm_generated.py`).
  - **London $N=61$ Polynomial Fitter** (`torchwave.qnm.london` / `torchwave-generate-london-qnm`): Solves the exact $N=61$ grid $a = 0.995\sin\theta$ and fits the clean-room rational polynomials for higher-harmonic ringdown models (`src/torchwave/approximants/imrphenom_hm_qnm_n61.py`).

### 2.13 IMRPhenomXPHM-SpinTaylor (Numerical Orbit-Averaged SpinTaylor Precession)
- **Primary References**:
  - Pratten et al. (2021), *Computationally efficient models for the dominant and sub-dominant harmonic modes of precessing binary black holes*, Phys. Rev. D 103, 104056 ([DOI:10.1103/PhysRevD.103.104056](https://doi.org/10.1103/PhysRevD.103.104056))
  - Racine (2008), *Analysis of spin precession in binary black hole systems including quadrupole-monopole interaction*, Phys. Rev. D 78, 044021 ([DOI:10.1103/PhysRevD.78.044021](https://doi.org/10.1103/PhysRevD.78.044021))
  - Kidder (1995), *Coalescing binary systems of compact objects to (post)5/2-Newtonian order. V. Spin effects*, Phys. Rev. D 52, 821–847 ([DOI:10.1103/PhysRevD.52.821](https://doi.org/10.1103/PhysRevD.52.821))
  - Blanchet (2014), *Gravitational Radiation from Post-Newtonian Sources and Inspiralling Compact Binaries*, Living Rev. Relativity 17, 2 ([DOI:10.12942/lrr-2014-2](https://doi.org/10.12942/lrr-2014-2))
- **Mathematical Form & Implementation**:
  - Orbit-averaged spin-precession equations for $\mathbf{S}_1, \mathbf{S}_2, \hat{\mathbf{L}}_N$ integrated in dimensionless velocity coordinate $v = (\pi M f)^{1/3}$ from PN energy flux $dv/dt = (c^3 / G M) \frac{32}{5} \eta v^9 [1 + \dots]$.
  - Precession Euler angles $\alpha(f)$ and $\beta(f)$ extracted from $\hat{\mathbf{L}}(f) = (\sin\beta \cos\alpha, \sin\beta \sin\alpha, \cos\beta)$, with minimal-rotation angle $\gamma(f)$ computed via $d\gamma/df = -\cos\beta (d\alpha/df)$.
  - Fixed-grid 4th-order Runge-Kutta (RK4) in $v$-space is expressed in PyTorch tensor operations. The model tests exercise batching and gradients for their declared parameters.
- **Verification Result**:
  The optional [LAL comparison test](https://github.com/xangma/diffgw/blob/main/tests/test_imrphenom_xphm_spintaylor.py) defines three configurations with `prec_version=320`, `final_spin_mod=2` and a minimum PyCBC time/phase-optimized plus-polarization match of $0.98$ over 20–300 Hz. This is a test gate, not an observed lower bound across the parameter domain or a direct complex-waveform error.

### 2.14 IMRPhenomXPNR (PhenomPNR Numerical Relativity Precession Angles)
- **Primary References**:
  - Hamilton et al. (2021), *The final twist: A model of gravitational waves from precessing black-hole binaries through merger and ringdown*, Phys. Rev. D 104, 124027 ([arXiv:2107.08876](https://arxiv.org/abs/2107.08876))
- **Mathematical Form & Implementation**:
  - Evaluates NR-calibrated opening angle $\beta(Mf)$, precession angle $\alpha(Mf)$, and minimal rotation angle $\gamma(f)$.
  - Maps two-spin binary parameters to effective single-spin parameters $(\chi, \cos\theta, \cos\theta_f, \chi_{\rm eff})$ via SpinTaylor trajectory at $v_{\rm ref}$.
  - Computes 11 polynomial calibration surfaces ($A_1..A_4, B_0..B_5, B_f$) in $(\eta, \chi, \cos\theta, \cos\theta_f, \chi_{\rm eff})$ via vectorized multi-linear tensor contractions (`src/torchwave/approximants/pnr_coefficients.py`).
  - Implements $C^1$ continuous quadratic rescaling $(1 + b_1 Mf + b_2 Mf^2)$ at the lower extremum $Mf_{\beta, \rm low}$ with analytical extremum root-finding.
  - Couples PNR precession angles to clean-room `IMRPhenomXHM` coprecessing carrier modes using Wigner $d$-matrices.
- **Verification Result**:
  The optional [LAL comparison test](https://github.com/xangma/diffgw/blob/main/tests/test_imrphenom_xpnr.py) defines four configurations against default LAL `IMRPhenomXPNR`, with a minimum PyCBC time/phase-optimized plus-polarization match of $0.99$ over 20–400 Hz. This threshold does not establish a full-domain accuracy bound.

---

## 3. Reference Evidence by Model

These counts describe the historical [catalog][frozen-catalog] and
[holdout][frozen-holdout] artifacts. A passing row is scoped to its recorded
parameters and modes. The complete strict suites did not pass; separate native
component tests do not change those results. The [current model manifest](model_manifest.json)
retains case IDs and numerical errors for supported routes; the
[retired snapshot](waveforms/research/retired-model-manifest.json) preserves
the removed model entries.

| Approximant | Catalog cases passed | Holdout cases passed | Scope |
|---|---|---|---|
| TaylorF2 | 4/4 | — | Declared aligned-spin/tidal cases |
| TaylorF2_Eccentric | 2/2 | — | Declared circular/eccentric cases |
| IMRPhenomD | 1/1 | 3/3 | Declared aligned-spin cases |
| IMRPhenomPv2 | 1/1 | 1/1 | Declared precessing cases |
| IMRPhenomXP | 1/1 | 1/1 | Supported MSA path; additional domain limits in manifest |
| IMRPhenomHM | 1/1 | 1/1 | Declared multimode cases |
| IMRPhenomXAS | 1/1 | 2/2 | Declared aligned-spin cases |
| IMRPhenomD_NRTidal | 1/1 | 1/1 | Declared tidal cases |
| IMRPhenomD_NRTidalv2 | 1/1 | 1/1 | Declared tidal cases |
| IMRPhenomNSBH | 1/1 | 1/1 | Declared NSBH cases |
| KerrRingdown | 1/1 | — | Canonical $(2,2), n=0$ physical-energy waveform |
| SEOBNR_ROM | 1/1 | — | Official external-data wrapper |
| IMRPhenomXHM | 0/1 | 0/1 | Full-waveform reference gates not met; native phase tests have separate scope |
| IMRPhenomXPHM | 0/1 | 0/1 | Full-waveform reference gates not met; native phase tests have separate scope |
| SEOBNR_ODE (retired prototype) | 0/1 | — | Historical result; model no longer shipped |
| TEOBResumS (retired prototype) | 0/1 | — | Historical result; model no longer shipped |
| TaylorF2_HM | 1/1 | — | Only $(2,2)$ compared; higher modes lack a full-waveform oracle in this suite |

`IMRPhenomXPHM-SpinTaylor` and `IMRPhenomXPNR` are outside these frozen
catalog counts. Their optional comparison test gates are described in Sections
2.13–2.14. Neither the frozen suite nor this table represents all current models.

---

## 4. Machine-Readable Asset Checksum Ledger

| Asset | Path | SHA256 Checksum | Provenance / Origin | License |
|---|---|---|---|---|
| Independent Kerr QNM Splines | `src/torchwave/approximants/kerr_qnm_generated.py` | `82da71c19c50defeab4c32b6950141d845214909e9f52342e3f1cf3b46f08326` | Clean-room solved via `torchwave.qnm.generate` | MIT |
| Independent London N=61 QNM Fits | `src/torchwave/approximants/imrphenom_hm_qnm_n61.py` | `350c958aef4e1554d5c2d6a6bc22f8b9bf3513c2a000f08dc3cb3de1cde05b06` | Clean-room solved via `torchwave.qnm.london` | MIT |
| PNR Calibration Coefficients | `src/torchwave/approximants/pnr_coefficients.py` | `e15b3fef15512d48873fa1728ffb446068bb38245ac2378d0eb8cb20d4e27355` | Clean-room derived from Hamilton et al. (2021) | MIT |
| APS IMRPhenomXAS Fits | Supplemental Material (Pratten et al. 2020) | `94b9334c19f3a7909864891ea278ab9a411e4f33b7b7e17cc01d927056d9ac52` | APS PRD Supplemental Material deposit | Redistribution terms unresolved; see third-party notices |
| External Official SEOBNRv4ROM HDF5 | Zenodo record 14999310 (`SEOBNRv4ROM_v3.0.hdf5`) | `4ff5317967d757fcad9323a74b75f4f63894ed87f4b1852d46cbe47cde488b9c` | Official public Zenodo release | Redistribution terms unresolved |
| Native SXS NR Calibrations | `src/torchwave/approximants/xhm_native_fits.py` | Native code module | Calibrated on SXS public BBH catalog | Software: MIT; source simulations: CC-BY-4.0 |

---

## 5. Third-Party Data Attribution

The native calibration surfaces in `torchwave` were trained on numerical relativity simulations produced by the **Simulating eXtreme Spacetimes (SXS) Project** (http://www.black-holes.org). These simulations are released under the Creative Commons Attribution 4.0 International (CC-BY-4.0) license. Complete data citations, dataset terms, and catalog IDs are documented in [**THIRD_PARTY_NOTICES.md**](../THIRD_PARTY_NOTICES.md).

---

## 6. Documented Differences from LALSimulation

Model-specific formulations, reference configurations and limitations are described on [Differences from LAL](differences_from_lal.md#model-specific-differences).

## Native IMRPhenomT family (2026-09-21)

- Physics: Estellés et al., Phys. Rev. D **103**, 124060 (2021),
  [arXiv:2004.08302](https://arxiv.org/abs/2004.08302),
  [DOI](https://doi.org/10.1103/PhysRevD.103.124060).
- Updated dominant-mode ansatz: Estellés et al.,
  [arXiv:2012.11923](https://arxiv.org/abs/2012.11923), Eqs. 14–31.
- Calibration and reconstruction reference: the MIT-licensed author repository
  [phenomxpy, pinned revision](https://gitlab.com/imrphenom-dev/phenomxpy/-/tree/4121fafa0484f3db2a7352f4eea697faaaf29f65/phenomxpy/phenomt).
  Shared calibration expressions include the supported `(2,2)`, `(2,1)`,
  `(3,3)`, `(4,4)` and `(5,5)` branches of `fits.py`; `internals.py` and `numba_ansatze.py` provide the reconstruction
  conventions. This is an attributed adaptation of permissive author math,
  not a claim that the numerical calibration coefficients were independently
  derived from the paper. The complete MIT notice is retained in
  [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
- Runtime: framework-native tensor operations and differentiable matching in
  both frontends, with backend-independent algebraic fits in `diffgw_core`.
  No LALSuite, SciPy or phenomxpy waveform calls are made at runtime.
- Scope: T, THM, TP and TPHM for black hole binaries. THM includes mode
  pairs `(2,2)`, `(2,1)`, `(3,3)`, `(4,4)` and `(5,5)`. TP/TPHM implement
  published NNLO, multiple-scale analysis (MSA), and numerical precession.
  Tidal corrections are absent.
- Precession reference: [arXiv:2105.05872](https://arxiv.org/abs/2105.05872).
  The native rotation uses published Euler-angle and Wigner-matrix formulas
  with the repository's NNLO tensor implementation. MSA uses the published
  two-spin equations of [Chatziioannou et al.](https://arxiv.org/abs/1703.03967)
  and the repository's existing clean-room algebra. Numerical precession uses
  the orbit-averaged spin equations through N4LO in
  [Sturani's technical note, LIGO-T1500554](https://dcc.ligo.org/public/0122/T1500554/023/dLdS.pdf),
  with the PhenomT frequency evolution. The numerical PhenomTP path omits
  the N3LO quadratic-spin contribution (Eq. 24), following the `phenomtp`
  parameter description in the
  [public LAL API](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_inspiral_8h.html).
  The complete published equations remain available in the internal spin
  derivative helper. Leading self-spin terms and the nonspinning 3PN
  angular-momentum denominator are retained; binary-oracle residuals alone
  do not justify altering those equations. The upstream phenomxpy numerical
  and MSA modules identify GPL-derived code and were excluded from adaptation.
- Verification: LALSimulation is used only through its installed binary
  Python bindings as an optional black-box oracle. No coefficients or
  residuals are fitted to those outputs, and no GPL implementation source
  is used. Backend and behavioral tests are in `tests/test_imrphenomt.py`,
  `tests/test_phenomt_public.py` and `tests/parity/test_phenomt_parity.py`.
  `tests/test_phenomt_reference.py` retains numerical values from the pinned
  MIT author implementation and optional LAL binary comparisons spanning
  equal masses, mass ratio 10 and spins from -0.9 to 0.9. Measured LAL
  residuals are recorded in [Differences from LAL](differences_from_lal.md#model-specific-differences).
  FP64 cross-backend tests require relative L2 error at most `1e-11` and
  mismatch at most `1e-14`, including the high-spin mass-ratio-10 JIT case.
  Phase-sensitive mass/spin gradients are checked against finite differences
  in Torch and against Torch derivatives in JAX.
  Higher-mode fixtures from the same pinned author release are retained in
  `tests/test_phenomthm_reference.py`; fit regressions are in
  `tests/test_phenomthm_fits.py`. Family dispatch, remnant-spin propagation,
  precession, and optional LAL comparisons have dedicated
  `test_phenomt_family_*`, `test_phenomt_remnant.py`, and
  `test_phenomt_precession.py` suites. Physical-time invariance under grid
  changes and time translations is checked in `test_phenomt_time_grid.py`. Cross-backend higher-mode and
  precession tests also cover JAX compilation and parameter gradients.

  The additional prescription tests cover method/version selection, matching
  remnant spins, batch behavior, reference-time anchoring, and invariance under
  changes to the requested output grid. The NNLO default is retained for
  compatibility; numerical precession is selected explicitly with
  `precession_method="numerical"`.
