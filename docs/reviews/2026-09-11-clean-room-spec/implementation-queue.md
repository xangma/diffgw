# Paper-based implementation queue for AGY

This specification separates identified implementation gaps from missing reference data. A model name and passing unit tests do not establish mathematical equivalence to that model. Preserve the existing strict catalog cases, thresholds, manifest inventory, and unaligned complex-strain comparisons.

## Implementation boundary

Use published mathematics, independently derived algebra, and independently published author calibration data with recorded URL/version/hash. Implement fresh PyTorch code. Do not read LAL/LALSimulation/PyCBC implementation source, internal coefficients, translated ports, old source-audit material, or oracle fixtures. Do not fit coefficients or phase/time offsets to oracle residuals. The verifier may compare public waveform outputs; the implementer receives behavioral requirements and diagnostic conclusions, not implementation excerpts. Missing author data must remain an explicit blocker.

The isolated implementation snapshot contains inherited TorchWave source; its historical provenance is not being certified here. Fresh conversation and limited inputs protect this phase, but are not an OS-enforced information barrier or a guarantee about earlier contributors.

## Ordered work packages

### 1. NRTidalv2 reference phase — first implementation

Use the supplied SPECIFICATION.md: expose the published 2PN/3PN and 3.5PN self-spin contributions separately. Anchor only the 2PN/3PN terms at the reference frequency under the specified compatibility convention. Preserve the existing coalescence behavior and v1. Equations and coefficients come from [Dietrich et al., Eqs. 26–29](https://arxiv.org/html/1905.06011v2); the split reference convention is a separately established behavioral requirement.

Acceptance: independent equation tests; complex tidal-to-BBH ratios; spin signs, body swaps, several reference frequencies; aliases and overrides; zero tides/spins; batch-vector reference frequencies; normalized full-complex gradient checks including reference frequency, DC and cutoff bins. Then run the frozen full-waveform catalog. Passing these local checks does not resolve PhenomD baseline or remaining tidal amplitude errors.

### 2. PhenomHM — reconstruct the published mapping

Observed local gap: imrphenom_hm.py uses inspiral frequency scaling throughout the phase, leading-order PN amplitudes, and a generic Lorentzian cutoff. Implement the separate piecewise amplitude/phase frequency maps, integrated phase connection and mapped baseline amplitude from [London et al., Methods, Eqs. 3–10](https://arxiv.org/html/1708.00404v4). Include the published spin-sensitive subdominant amplitude terms and investigate the missing (4,3) mode; the paper lists six positive-m modes while the current default has five. Keep PhenomHM and PhenomXHM as distinct model constructions.

Acceptance: independent mapping values and one-sided join limits, phase derivative connections, low-frequency mode limits, unprojected mode normalization, mode-by-mode and summed polarization checks, body relabeling, equal-mass unequal-spin systems, all advertised modes, finite parameter gradients. Baseline PhenomD precision remains a dependency for strict parity.

### 3. PhenomXHM — implement its own calibrated higher modes

Observed local gap: IMRPhenomXHM merely changes the carrier in the HM class. Replace this with the calibrated inspiral/intermediate/ringdown amplitude and phase construction for (2,1), (3,3), (3,2), and (4,4), including (3,2)/(2,2) spherical–spheroidal mixing. [García-Quirós et al., Secs. IV–VI and Appendix A](https://arxiv.org/html/2001.10914v2) and the author's [ancillary mathematical release](https://arxiv.org/src/2001.10914v2/anc) are the sources. A previous research download found 91 ancillary assets; locate and hash the relevant coefficient assets before implementation.

Do not force every odd-m mode to zero solely because the masses are equal: the current code does that even with unequal spins. Derive the required symmetry including spin differences from the paper's parameter-space construction. Acceptance: coefficient provenance, paper collocation residuals, joins and gradients, individual modes before projection, mixing limits, and equal-mass unequal-spin tests. Global XAS time calibration remains unresolved; mode reconstruction can proceed while that dependency is documented.

### 4. XP/XPHM — complete precession prescriptions and conventions

Observed local gap: XPHM offers NNLO angles or an ODE path but no MSA implementation. Implement the two-spin multiple-scale-analysis prescription and its initialization/fallback rules from [Pratten et al., Sec. IV.2, appendices, Tables 3–5](https://arxiv.org/html/2004.06503v2), together with its cited primary derivation. The paper's default is MSA variant 223; the local default is NNLO. An ODE option is not equivalent to that analytic prescription. Also audit the documented angular-momentum order, reference-frame offsets, final-spin prescription and polarization rotation.

Acceptance: declared convention/version mapping, aligned-spin limit, two independent in-plane spins, body exchange, reference-frequency changes, mode projection, and normalized complex gradients. Pin external comparator options explicitly. XP depends on XAS; XPHM also depends on the completed XHM modes. Pv2 needs its own frame/angle comparison and must not inherit an MSA requirement merely by association.

### 5. PhenomNSBH — replace the schematic amplitude

Observed local gap: imrphenom_nsbh.py applies a simple disruption window to a BNS tidal waveform. Implement the actual [Thompson et al. model, Sec. II and Appendices A/B](https://arxiv.org/html/2002.08383v2): PhenomC-based component amplitude, relativistic disruption root, torus fraction, NSBH remnant fits, merger classification and calibrated joining windows; use its separately specified tidal phase. Acquire the cited remnant supplementary coefficients before completing dependent code. Reconcile the low-deformability/BBH limit and the supported nonspinning-NS convention.

Correct the module's attribution: Matas arXiv:2004.10001 describes SEOBNR_NSBH, a different model. Pannarale's amplitude paper is [arXiv:1509.00512 / PRD 92, 084050](https://arxiv.org/abs/1509.00512); arXiv:1509.06209 is the related cutoff-frequency paper, PRD 92, 081504. Verify manifest citations without changing parity gates.

Acceptance: independent root residuals and branch selection, each disruption regime, compactness interpolation, remnant limits, waveform component joins, zero-tide behavior under the target model's convention, supported spins, scalar/batched agreement and gradients away from physical branch boundaries.

### 6. Reference-data blockers and residual tidal work

- **PhenomD:** obtain an independently published unrounded author calibration release. Existing printed Table V values do not uniquely determine strict-tolerance behavior under the documented rounding sensitivity. This is not proof that every remaining residual is rounding. Do not transplant LAL-only extra digits.
- **XAS and descendants:** locate the original dominant-mode time/peak calibration discussed in XHM Appendix C. The previously retrieved XAS and XHM mathematical assets did not identify that coefficient set. Relative higher-mode timing fits are not substitutes. Do not regress a phase slope from the oracle and put it in production.
- **NRTidal amplitude:** maintain the explicitly derived strain/mode convention. Eq. 24 is an additive face-on strain correction; a mode conversion must be applied consistently to the entire projection. Do not add a scalar just because it lowers oracle error. Remaining amplitude and v1 residuals require further equation/public-output diagnosis with the base model separated.

For every unavailable asset record exactly what was searched and what remains missing. Search exhaustion is not proof of nonexistence. Data acquisition can run independently of the implementation order above.

### 7. Verification and coverage closure

The manifest currently has five unresolved oracle mappings: enumerate them from the manifest, specify an independent validation route for each, and retain unresolved status until evidence exists. Broaden physical parameter holdouts after component checks pass. Refresh stale descriptive measurements/citations only from recorded results; never weaken or relabel the gates to obtain a pass.

Every package ends with a reviewed diff, source provenance, native-runtime dependency check, relevant tests and the unchanged strict catalog. A failed strict catalog remains a reported failure even when a package improves the waveform. No all-model completion claim is authorized by a passing component test.
