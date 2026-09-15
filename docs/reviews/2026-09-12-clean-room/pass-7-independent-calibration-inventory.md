# Independent calibration inventory

Reviewed 2026-09-12. Scope: the current native TorchWave implementation and
primary papers; no LAL/PyCBC implementation, translated port, oracle output,
external fitting program, or serialized surrogate was used to derive data.
This report changes no source code. TorchWave was absent from the available
indexed-project list, so discovery used local source reads and AST inspection.

**Self-generated Kerr data does not make the complete waveform calibration
self-generated.** The models below still contain empirical waveform, remnant,
or matter fits. Published equations can be implemented independently while
retaining attribution; reproducing their printed fit constants is a separate
claim from obtaining new constants from our own simulations. Rounding an
inherited number, or fitting a native model that already uses that number,
does not establish independent numerical provenance.

On 2026-09-13 the user chose to **use the existing calibration tables for now**
and defer further NR work. The [execution record](pass-8-len-nr-pilot.md) records
the stopped 48-worker performance run and preserved artifacts. The inventory
below still distinguishes published coefficients from inherited values with
unresolved extra-digit provenance. This operating choice does not establish new
provenance for those values. The large NR campaign below remains a future route
where new calibration coefficients are wanted.

## Dependency inventory

Paths below are relative to `src/torchwave/approximants/` unless stated otherwise.
Counts describe source definitions, not a proof that every definition is used
by every runtime option.

| Component and exact native location | What remains | Independent replacement |
| --- | --- | --- |
| PhenomD: `phenomd_unrounded_fits.py`, `TABLE_V_UNROUNDED_COEFFS`; consumers in `imrphenomd.py` | 19 parameter rows, 11 values each: 209 inherited values for `rho1–3`, `v2`, `gamma1–3`, `sigma1–4`, `beta1–3`, `alpha1–5`. The module explicitly leaves the extra-digit provenance unresolved. | Implement the Table V/Eq. (31) parameterization using independently transcribed printed values for a documented paper-precision version; for **our own calibration**, build and fit our own inspiral/NR hybrid waveforms, with held-out NR validation. The original construction used 19 calibration NR waveforms, a scale reference rather than a sufficient new design. [Khan et al., Secs. III–VIII and Table V](https://arxiv.org/html/1508.07253v2). |
| PhenomD remnant: `phenomd_unrounded_fits.py`, `FINAL_SPIN_ETA_POLY`, `FINAL_SPIN_S_COEFFS`, `ERAD_ETA_NUMERATOR`, `ERAD_S_NUMERATOR`, `ERAD_S_DENOMINATOR`; `IMRPhenomD.remnant_properties` | Empirical final-spin/radiated-energy expressions, including inherited extra digits. QNM replacement does not replace these binary-to-remnant maps. | Fit final horizon mass/spin or consistently extracted energy/angular-momentum losses from our own BBH simulations, enforcing appropriate test-mass constraints. Printed Eqs. (3.6)–(3.8) provide a published-fit implementation route. [Husa et al.](https://arxiv.org/abs/1508.07250). |
| PhenomHM: `imrphenom_hm.py`, `IMRPhenomHM`, `mode_qnm_frequencies`, `QNM_FITS` | Default D carrier, D remnant dependency, and Berti spectral fits; optional XAS carrier adds XAS dependencies. Its PN amplitude/frequency mappings are distinct from the carrier's NR fit. | Reuse independently implemented PN mappings and regenerated Kerr spectra. No separate HM NR calibration is required by this model's construction; a new carrier still requires the work above, and HM accuracy needs independent NR validation. [London et al.](https://arxiv.org/abs/1708.00404). |
| PhenomXAS: `imrphenom_xas.py`, `XASParameterFits`, `LINB_*`, `PSI4_*`; bundled `src/torchwave/data/IMRPhenomXAS_Parameter_Space_Fits_Public_Release_Candidate.m` | 27 fit methods: 22 waveform fits, two remnant fits, two Kerr fits and one MECO fit; additional fitted time/phase alignment arrays. The native header attributes expressions to an author Mathematica asset, which is an inherited numerical source, not our generated calibration. | Regenerate Kerr quantities; solve hybrid MECO directly as described below. New waveform/remnant fits require our own aligned-spin hybrids, unequal-spin coverage and large-mass-ratio perturbative waveforms. Independently implementing the author's published fit remains a different option. [Pratten et al.](https://arxiv.org/html/2001.11412v1). |
| PhenomXHM: `xhm_fits.py`, `XHMParameterFits`; `imrphenom_xhm.py` | 37 amplitude methods and 35 phase methods, plus 10 QNM methods and four angular-mixing methods. This includes calibrated `(3,2)` spheroidal phase/derivative and relative-time/phase information, and XAS carrier/remnant dependencies. Existing bounded checks do not certify the entire ancillary coefficient collection. | Generate Kerr spectra and angular overlaps independently; fit the remaining mode amplitudes/phases against our own `(2,1),(3,3),(3,2),(4,4)` hybrids, including consistent mode phase conventions and test-mass waveforms. Angular overlaps alone do not supply the merger excitation or relative-phase calibration. [García-Quirós et al.](https://arxiv.org/abs/2001.10914). |
| NSBH: `imrphenom_nsbh.py`, `ns_compactness_from_lambda`, `bbh_remnant_uib2016`, `zappa_remnant_properties`, `london_ringdown_properties`, and `IMRPhenomNSBH.forward` | UIB BBH remnant values, Zappa NSBH remnant fit, compactness–deformability fit, torus/disruption/window parameters, PhenomC `gamma1/delta1/delta2` amplitude fits, London QNM polynomial, D phase and NRTidal v2. `_legacy_phase_offsets` is an inherited gauge prescription, not a numerical-relativity dataset. | BBH remnant simulations plus NSBH general-relativistic hydrodynamics across mass ratio, BH spin and neutron-star structure; extract disk mass, disruption and waveform/remnant observables. Direct stellar-structure/tidal calculations can replace compactness fits, but require an explicit equation of state. [Thompson et al.](https://arxiv.org/abs/2002.08383), [Pannarale et al.](https://arxiv.org/abs/1509.00512), [Zappa et al.](https://arxiv.org/abs/1903.11622). |
| NRTidal: `nrtidal_unrounded_fits.py`, `NRTIDAL_V1_PADÉ_COEFFS`, `NRTIDAL_V2_PHASE_COEFFS`, `NRTIDAL_V2_AMP_COEFFS`, `BNS_MERGER_FREQ_COEFFS`; `imrphenomd_nrtidal.py`, `_nrtidal_self_spin_components` | Tidal phase/amplitude and merger-frequency calibration, plus EOS-dependent multipole universal relations. Some constants are exact PN fractions; other decimals and the noninteger amplitude power are empirical. The blanket clean-room/MIT provenance header is not substantiated by this inventory. | Refit tidal residuals using our own BNS numerical waveforms and independent PN/EOB inspirals; retain exact analytic terms separately. Generate stellar tidal/multipole relations from a documented EOS family. [NRTidal v1](https://arxiv.org/abs/1706.02969), [NRTidal v2](https://arxiv.org/abs/1905.06011), [merger-frequency model](https://arxiv.org/abs/1804.02235). |
| Published EOB components: `eob/seobnrv4_hamiltonian.py`, `calibration_parameters`; `eob/nqc.py`, `seobnrv4_nqc_targets`, `seobnrv4_nqc_time_offset`; `eob/merger_ringdown.py`, `seobnrv4_merger_shape` | Explicitly printed empirical `K`, `dSO`, `dSS`, peak targets/time offset, and free merger-shape constants. These were independently reconstructed from papers; they are still the authors' NR/perturbative calibration. Solving NQC matching equations does not independently generate the targets. | Calibrate dynamics and peak/matching data against our own BBH and test-particle waveforms, then fit the parameter dependence. [Bohé et al., Secs. II, IV and Appendix A](https://arxiv.org/abs/1611.03703). |
| EOB remnant: `eob/remnant.py`, `_HOFMANN`, `hofmann_final_spin`, `eob_final_mass_fraction` | Printed Hofmann spin coefficients and Ossokine radiated-energy constants. The Kerr ISCO calculation is analytic; the binary remnant maps are empirical. | Share an independently generated BBH remnant dataset with D/XAS/NSBH baseline work. [Hofmann et al., Table 1](https://arxiv.org/abs/1605.01938), [Ossokine et al., Eq. (3.5)](https://arxiv.org/abs/2004.09442). |
| Native ROM: `rom/generator.py`, `generate_eob_training_grid`, `train_and_export_surrogate`; `rom/surrogate.py`, `_get_native_surrogate`; `src/torchwave/data/seobnr_surrogate.pt` | Current generator calls native `SEOBNR_ODE` and writes an MIT metadata label. The supplied weight artifact's actual generation lineage was not inspected or executed. The current teacher is documented as a prototype. | Regenerate from an independently verified native teacher; record generator/source hashes, options, training grid, tolerances, splits and residuals. This creates our own approximation of that teacher; it does not remove the teacher's calibration or establish SEOBNRv4 fidelity. |
| TEOB prototype: `teobresums.py`, tidal amplification/NS quadrupole defaults and `compute_tidal_merger_frequency` | Source explicitly marks inherited tidal/merger heuristics and prototype taper behavior. These are not established published TEOBResumS calibration assets. | Replace heuristics with a specified published model or new calibration, using stellar-structure calculations and BNS simulations as applicable. Re-labeling the heuristic or sampling it does not produce a validated tidal merger fit. |

## Work that can be done without binary merger simulations

1. **Kerr spectra and angular overlaps:** solve the radial/angular perturbation
   problems and generate interpolation assets. The root task owns this work.
2. **XAS hybrid MECO:** independently minimize the PN/Kerr hybrid energy rather
   than reuse `fit_f_meco`. This fit approximates a root-finding problem, not an
   NR waveform dataset. Select the PN order and spin conventions explicitly;
   verify the derivative residual, branch and test-particle limit.
   [Pratten et al., Eqs. (48)–(49), Sec. V.1](https://arxiv.org/html/2001.11412v1).
3. **Analytic waveform machinery:** PN coefficients, spherical harmonics,
   continuity/matching linear systems, units, Fourier conventions and integration
   algorithms can be implemented and tested from equations. A pseudo-PN term
   fitted to hybrids is not an exact PN coefficient.
4. **Stellar relations:** solve equilibrium/tidal perturbation equations for a
   specified EOS instead of importing a universal-relation fit. This supplies
   isolated-star quantities; binary merger/disruption still needs dynamics.

The earliest missing-data route is therefore Kerr spectra/overlaps, followed by
direct hybrid-MECO solutions and reproducible native-teacher ROM generation.
For D, independently transcribe the rounded paper tables if that precision is
acceptable. Missing original extra digits cannot be recovered from the printed
tables; obtaining new fitted digits would create a new calibration, not recover
the original one. Published EOB peak/remnant fits can remain as explicitly cited
mathematical inputs under the narrower scope.

## Concrete route to entirely new calibrations

Start with one declared aligned-spin BBH domain and the dominant mode. Build a
reproducible NR pipeline: initial data and eccentricity control, spacetime
evolution, horizon diagnostics, mode extraction and asymptotic extrapolation.
Establish convergence with multiple resolutions and extraction choices before
using a waveform as calibration data. Combine those data with an independently
specified PN/EOB inspiral, record the hybridization interval and errors, then fit
the waveform ansatz and remnant surfaces. Reserve configurations for validation
before choosing the fit; LAL output must not supply training targets or tune
coefficients. Evaluate mode errors and detector-weighted mismatches separately.

Extend the same simulation products to higher modes and unequal spins, and add
test-particle evolution for large mass ratios. Extend to neutron stars only
after choosing matter/EOS assumptions and validating hydrodynamic evolution.
The UIB remnant paper used 427 NR simulations, illustrating that accurate
three-parameter remnant coverage is substantive work, not a small spectral
lookup replacement. This is a historical dataset size, not a resource estimate
for our campaign. [Jiménez-Forteza et al.](https://arxiv.org/abs/1611.00332).

Training a surrogate on a native calibrated waveform can be useful immediately,
but its metadata must identify that dependency. A newly fitted waveform must
also carry its own version and validation domain: it will not reproduce the
authors' coefficients exactly simply because it uses the same ansatz.

For every new numerical asset, record equations, generator commit/source hash,
inputs, numerical tolerances, training/validation partition, convergence/error
results, output checksum and ownership/provenance. Retain literature attribution
and isolate inherited extra-digit arrays until independently replaced or their
source established. No compute-time claim is made here: the repository has not
demonstrated a converged end-to-end NR calibration campaign or a benchmark from
which to estimate one.

## Verification and limits

Read the listed native definitions and dependencies; counted fit definitions
with Python AST without importing approximants. Checked the cited primary
paper records and the relevant PhenomD/XAS construction sections. No oracle
calls, fitting, binary deserialization, training or remote jobs were performed.
This is a bounded dependency inventory, not a complete numerical validation or
an assertion that every inherited decimal has been independently sourced.
