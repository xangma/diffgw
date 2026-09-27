# Higher-mode, eccentric, and tidal waveform family: research ledger

This family research ledger began at foundation `5bc907d`. It follows the
[six-section specification template](../../math/waveform-spec-template.md).
`TaylorF2_HM` has a [versioned leading-carrier specification](../../math/taylorf2-hm.md),
and `TaylorF2_Eccentric` has a [versioned nonspinning amplitude-off slice](../../math/taylorf2-eccentric.md).
The remaining three rows remain provisional. The component IDs below are historical
research identifiers, not registered catalog IDs.

## Scope and status

| Native approximant | Implemented mathematical scope | Research status |
| --- | --- | --- |
| `TaylorF2_HM` | Frequency-domain, aligned-spin SPA inspiral, leading amplitudes for `(2,2), (2,1), (3,1), (3,2), (3,3), (4,2), (4,3), (4,4)` | [Independent leading-carrier reference](../../math/taylorf2-hm.md) covers all eight modes; exact external eight-mode waveform comparison remains unavailable. |
| `TaylorF2_Eccentric` | Frequency-domain `(2,2)` quasi-circular TaylorF2 plus an order-`e0²` eccentric phase through relative 3PN; optional project amplitude factor | The [nonspinning amplitude-off composition](../../math/taylorf2-eccentric.md) has an independent scalar reference and two strict LAL holdouts in Torch/JAX; spins and optional amplitude remain provisional. |
| `IMRPhenomD_NRTidal` | Frequency-domain aligned-spin `(2,2)` IMRPhenomD plus 2017 NRTidal phase and BNS merger taper | The [nonspinning tidal-phase component](../../math/nrtidalv1-phase.md) passes independent scalar and pinned-binary quotient gates; merger/taper, spins, and the whole waveform remain provisional. |
| `IMRPhenomD_NRTidalv2` | Same BBH carrier plus 2019 phase, historical default tidal amplitude, EOS-dependent self-spin, and merger taper | [Nonspinning phase](../../math/nrtidalv2-phase.md), [amplitude/taper](../../math/nrtidalv2-amplitude-taper.md), and [spinning paper-D composition through taper](../../math/nrtidalv2-paper-d-composition.md) have scoped independent references. The paper gate follows Eqs. (28)–(29) despite their Fig. 7 conflict; the historical default amplitude and high-frequency pinned-binary taper retain provenance and binary gaps. Spinning whole-waveform qualification remains pending. |
| `IMRPhenomNSBH` | Frequency-domain aligned-spin `(2,2)` PhenomD-like phase with NRTidalv2 term and a disruption-dependent PhenomC-like amplitude | [Compactness/disruption-root](../../math/nsbh-disruption-root.md) and [torus/ISCO](../../math/nsbh-torus-isco.md) components pass independent scalar paper gates. Amplitude branches, remnant, phase, and the whole waveform remain provisional. |

Torch and JAX expose all five through their respective `get_fd_waveform`
dispatchers. `TaylorF2_HM` and the narrow `TaylorF2_Eccentric` slice are registered in
`src/diffgw_core/waveform_definitions/catalog.py`; the other three are not. The historical
[`model_manifest.json`](../../model_manifest.json) statuses describe *declared*
CPU FP64 waveform comparisons, not independently executable scalar expressions
for the remaining provisional models or a parameter-domain certificate. This ledger's
foundation audit did not rerun comparisons; the HM specification links its
subsequent maintained checks.

## Inputs, conventions, and options

All five native frontends accept gravitational-wave frequency grids in Hz
(including a possible DC bin), component masses in solar masses, distance in
Mpc, aligned dimensionless spins, inclination, orbital coalescence phase, and
coalescence time. They return complex `h_plus`, `h_cross`; nonpositive-frequency
bins are zeroed by the assembled models. The phase is placed in an
`exp(-i Ψ)` factor. Existing TaylorF2 and IMRPhenomD documents define the
carrier conventions: [TaylorF2](../../math/taylorf2.md) and
[IMRPhenomD](../../math/imrphenomd-waveform.md). The exact sign of a composed
mode and projection must be included in each eventual standalone spec.

| Model | Additional native inputs/defaults and convention requiring a reference case |
| --- | --- |
| HM | `modes` defaults to all eight; `return_modes` returns per-mode polarizations; `phi0=0`; `f_isco_cutoff=True`, Schwarzschild or Kerr `isco_type`, optional smooth rolloff of width `roll_off_delta=0.1`; `spin_order=7`; `f_ref`, `lambda1/2` pass through to TaylorF2. Frequency and cutoff for mode `m` are rescaled by `m/2`. |
| Eccentric | `e0`/`eccentricity` aliases (manifest quotes `[0,0.4)`); `f0_ecc` takes precedence over `f_lower`, then positive `f_ref`, then the lowest positive grid sample (20 Hz if none). Eccentricity is defined at that frequency; an explicitly supplied positive `f_ref` independently anchors the waveform phase, including the eccentric correction. `ecc_order=6` means 3PN *phase*, not order `e0^6`; `include_amplitude_correction=False`. Spin aliases and `phasing_convention`/`reference_frame` are forwarded to TaylorF2. |
| NRTidal v1/v2 | `lambda1/2` default to zero; `apply_merger_taper=True` applies a Planck window from `f_merger` to `1.2 f_merger`. `f_ref`, `phasing_convention`, and `reference_frame` pass to PhenomD. For v2 reference-frequency phasing, the native code subtracts the 2PN+3PN self-spin term at `f_ref` but retains the 3.5PN term. This is a carrier choice needing its own comparison. |
| NSBH | `mass1` is BH, `mass2` is NS, `lambda2` is the NS deformability; source calibration assumes negligible NS spin. Paper scope is `1≤q≤15`, `|chi_BH|≤0.5`, `0≤Lambda_NS≤5000` ([Thompson et al.](https://arxiv.org/pdf/2002.08383v1), abstract), narrower than some accepted native inputs. `f_ref` defaults to 30 Hz; explicit zero selects the lowest positive sample. Default/`legacy` phase has inherited affine offsets, including an upper-grid-frequency anchor unless `f_final`/`fhigh` is supplied. Explicit `coalescence` or `reference_frequency` selects the PhenomD phase gauge. |

The HM and eccentric models are inspiral-only approximants with cutoff
adapters; the NSBH and NRTidal models inherit calibrated IMR carrier data.
There is no separately packaged training asset for this family; fit decimals
are embedded in the cited Python modules. Those decimal rows and all branch
boundaries require versioned provenance in canonical specs. This ledger has
not established derivative behavior at the hard HM cutoff, NSBH
classification boundaries, or a failed NSBH disruption-root iteration.

## Mathematics and components

Let `B_TF2` and `B_D` denote the existing TaylorF2 and IMRPhenomD carriers,
`x=(πMf)^(2/3)`, `η=m1 m2/M²`, and `κ_eff=(3/16) Λ̃`. The native compositions
are:

* HM: `h_{l,-m}(f) = sqrt(16π/5) A_TF2(f) R_lm(f)
  exp[-i{(m/2) Ψ_TF2(2f/m) + 3π(1-m/2)/4}]`, followed by the implemented
  spin-weighted-harmonic projection and per-mode cutoff.
* Eccentric: `Ψ = Ψ_TF2 + ΔΨ_ecc(f;f0) - ΔΨ_ecc(f_ref;f0)`, with
  `ΔΨ_ecc = -(2355/1462)[3/(128ηv⁵)] e0²(v0/v)^(19/3) P_3PN(v,v0,η)`.
  The optional amplitude multiplier is `1-(743/9216)e0²(f/f0)^(-19/9)`;
  its source attribution is unresolved here.
* NRTidal v1/v2: `h = B_D exp(-i Ψ_T) W_merger` for v1, with
  `Ψ_T=-κ_eff[39/(16η)]x^(5/2)P_v1(x)`. V2 replaces `P_v1` by
  `P_v2`, adds 2PN/3PN/3.5PN EOS self-spin phasing, and applies an
  additive tidal amplitude before the window. The native v2 implementation
  converts that face-on amplitude into a common polarization multiplier.
* NSBH: `A_NSBH = A_PN W_ins + A_PM W_pm + A_RD W_rd` with branch-dependent
  joins; `Ψ_NSBH = Ψ_PhD + Ψ_NRTidalv2`. Native synthesis applies the
  quadrupolar inclination factors and its chosen affine phase gauge.

`pending` means that a paper-literal independent scalar expression and its
specified conventions are still needed. `conditional` means a reusable
reference exists but additional source or carrier inputs must be qualified.
`diagnostic` denotes a check, never production mathematics. Native functions
in the table already execute; status here describes the *library reference*
work, not native availability or verification.

| Research component ID/version | Pinned source and variant | Status | Independent scalar / native mapping and blocker |
| --- | --- | --- | --- |
| `hm.mode-phase/v0` | [García-Quirós et al. 2020](https://doi.org/10.1103/PhysRevD.102.064002), Eqs. (4.8)–(4.10), inspiral harmonic rescaling; [Mezzasoma & Yunes 2022](https://arxiv.org/abs/2203.15934), Eqs. (2.24)–(2.30), SPA phase and offset | superseded in scope | The [v1 carrier](../../math/taylorf2-hm.md) checks aligned-spin phase and both phase origins for all eight modes. Tidal terms remain outside that HM slice. |
| `hm.leading-amplitudes/v0` | [Kidder 2008](https://arxiv.org/abs/0710.0614), [García-Quirós 2020](https://arxiv.org/abs/2001.10914), [Mezzasoma & Yunes 2022](https://arxiv.org/abs/2203.15934), Eqs. (2.20)–(2.21i), especially `(4,2)` Eq. (2.21g); leading PN only | superseded in scope | The [v1 carrier](../../math/taylorf2-hm.md) derives all eight leading ratios and compares complex projected modes with Torch/JAX. Higher-PN and spin-amplitude terms remain outside scope. |
| `hm.projection-cutoff/v0` | García-Quirós Eqs. (2.1)–(2.12) for mode and polarization conventions; project harmonic ISCO adapter | superseded in scope | The [v1 carrier](../../math/taylorf2-hm.md) checks negative-`m` projection, azimuth derivatives, and Schwarzschild hard-cutoff thresholds; Kerr and smooth cutoff options remain unchecked. |
| `ecc.phase-3pn-e2/v1` | [Moore et al. 2016](https://arxiv.org/pdf/1605.00304v2), **Eq. (6.26)**, TaylorF2 SPA phase at order `e0²` | scoped executable | Independent SymPy rational polynomial and 80-digit scalar evaluation check all six native truncations in Torch/JAX CPU float64; distinct `f_ref` subtraction is the separately specified project gauge. See [component specification](../../math/taylorf2-eccentric-phase.md). |
| `ecc.amplitude-adapter/v0` | Moore et al. Sec. VI.E uses 0PN amplitude at order `e0⁰`; native `743/9216` factor is a project option | pending | Torch/JAX `eccentric_amplitude_factor`; identify a primary derivation or mark explicitly project-only; default is off. |
| `nrt1.phase-pade/v0` | [Dietrich, Bernuzzi & Tichy 2017](https://arxiv.org/abs/1706.02969), Eq. (7), 2017 rational phase | pending | `diffgw_core.nrtidal_fits.NRTIDAL_V1_PADÉ_COEFFS`; Torch/JAX `imrphenomd_nrtidal` v1 branch. Need independent scalar and paper-decimal precision comparison. |
| `nrt2.phase-pade/v0` | [Dietrich et al. 2019](https://arxiv.org/abs/1905.06011), Eqs. (17)–(21), frequency-domain phase, constrained coefficients | superseded in scope | The [versioned nonspinning phase](../../math/nrtidalv2-phase.md) checks the published Padé form and Torch/JAX below merger; the NSBH composition and self-spin remain outside that gate. |
| `nrt2.tidal-amplitude/v0` | Dietrich 2019, Eqs. (23)–(25), fitted `d=13477.8` | superseded in scope | The [Eq. (24) amplitude component](../../math/nrtidalv2-amplitude-taper.md) is independently checked, but Table II excludes it from the paper's D option. The historical default's extra `1/2` normalization remains without paper provenance; see the [audit](../../validation/nrtidalv2_provenance_20260927.md). |
| `nrt2.self-spin-eos/v1` | [Dietrich 2019, Eqs. (26)–(29), Table II](https://arxiv.org/html/1905.06011) | conditional | [Printed-equation D spec](../../math/nrtidalv2-self-spin.md), scalar reference, Torch/JAX gates and [receipt](../../validation/nrtidalv2_self_spin_20260927.md) cover positive-Lambda 2PN+3PN. Fig. 7 conflicts with Eqs. (28)–(29); the historical fit, Lambda clamps, anchoring and spinning binary gate are separate. |
| `nrt2.paper-d-composition/v1` | [Dietrich 2019, Eqs. (17)–(21), (25)–(29), Table II](https://arxiv.org/html/1905.06011); [Dietrich 2018, Eq. (11)](https://arxiv.org/pdf/1804.02235) | conditional | [Spinning complex-composition gate](../../math/nrtidalv2-paper-d-composition.md) and [receipt](../../validation/nrtidalv2_paper_d_composition_20260927.md) join the independent phase, self-spin and taper references through the merger boundary for Torch/JAX. Paper Fig. 7 and pinned-LAL amplitude/merger conflicts prevent a stronger binary or physical claim. |
| `nrt.merger-taper/v0` | [Dietrich et al. 2018](https://arxiv.org/abs/1804.02235), Eq. (11) BNS frequency; [McKechan et al. 2010](https://doi.org/10.1088/0264-9381/27/8/084020) taper; project `1.2 f_merger` endpoint | superseded in scope | The [scalar taper component](../../math/nrtidalv2-amplitude-taper.md) checks printed equations and backend behavior. Its nonzero-tide high-frequency binary quotient exceeds the frozen gate; [source audit](../../validation/nrtidalv2_provenance_20260927.md) narrows the unresolved boundary. |
| `nsbh.compactness-disruption/v0` | [Thompson et al. 2020](https://arxiv.org/pdf/2002.08383v1), Appendix A Eqs. (8)–(10), Appendix B Eq. (32) | superseded in scope | The [v1 independent scalar gate](../../math/nsbh-disruption-root.md) checks Eq. (32) and Eqs. (8)–(10), including largest-positive-root selection and normalized residuals, on a frozen grid. Branch use and whole-waveform behavior remain pending. |
| `nsbh.torus-isco/v1` | [Thompson et al. 2020](https://arxiv.org/pdf/2002.08383v1), Appendix A Eqs. (11)–(12) | scoped executable | The [independent scalar gate](../../math/nsbh-torus-isco.md) checks the raw baryonic torus fit, the Kerr ISCO radius, and the native nonnegative clip on 105 frozen cases in Torch/JAX CPU FP64. This does not qualify merger-classification branches or waveform composition. |
| `nsbh.remnant-qnm/v0` | Thompson Appendix A Eqs. (13)–(19); [Zappa et al. 2019](https://arxiv.org/abs/1903.11622), [Jiménez-Forteza et al. 2017](https://arxiv.org/abs/1611.00332), [London et al. 2018](https://arxiv.org/abs/1708.00404) | conditional | Torch/JAX `bbh_remnant_uib2016`, `zappa_remnant_properties`, `london_ringdown_properties`; paper/supplement coefficient provenance and inherited PhenomD remnant relationship need audit. Existing PhenomD scalar remnant is reusable only after variant alignment. |
| `nsbh.amplitude-branches/v0` | [Pannarale et al. 2015](https://doi.org/10.1103/PhysRevD.92.084050) and Thompson Eq. (4), Table I, Appendix A Eqs. (20)–(30) | pending | Torch/JAX `imrphenom_nsbh` amplitude assembly, `_amplitude_window`; Torch helper `nsbh_clean_amplitude`, JAX local equivalents. Reproduce every disruptive/non-disruptive branch, half-width typo correction, and `(Λ≤1)` interpolation. |
| `nsbh.phase-gauge/v0` | Thompson Sec. II.B PhenomD + NRTidalv2 phase; project legacy affine gauge | conditional | Existing `diffgw_reference.imrphenomd_phase` is a partial base. Torch `NSBHPhaseBaseline`, `_legacy_phase_offsets`, `nsbh_clean_phase`; JAX `imrphenom_nsbh` analogous path. Separate intrinsic phase from default grid-dependent affine offsets and qualify NSBH remnant provider. |
| `family.historical-comparison/v0` | Frozen manifest/provenance-ledger LAL comparisons, declared samples only | diagnostic | No production reference component; case IDs and recorded metrics in `docs/model_manifest.json`. Rerun only after an independent component reference is ready. |

## Implementation mapping

The table names the native entry points. The independent
`diffgw_reference.taylorf2_hm` module covers the declared leading HM carrier;
`diffgw_reference.taylorf2_eccentric` composes the bounded nonspinning
amplitude-off eccentric slice. No complete scalar reference exists yet for
the remaining three models. Existing
scalar TaylorF2 nonspinning, aligned-spin, tidal, spin-quadrupole and
IMRPhenomD amplitude/phase/remnant/intrinsic modules can be reused as
qualified carrier components, but they cannot be treated as references for
the remaining added terms. Native Torch/JAX evaluators do not call these
scalar modules.

| Family | Torch location | JAX location | Carrier/reference reuse and specific divergence to resolve |
| --- | --- | --- | --- |
| HM | `src/torchwave/approximants/taylorf2_hm.py` (`_swsh_projection_coefficients`, `TaylorF2_HM.forward`) | `src/jaxwave/approximants/taylorf2_hm.py` (same projection, `evaluate`) | `TaylorF2.evaluate_phase/amplitude`; stored `h_{l,-m}` and harmonic-specific cutoff are project composition choices. `(4,1)` and `(5,5)` remain absent because TD amplitudes alone do not fix native SPA/projection conventions. |
| Eccentric | `src/torchwave/approximants/taylorf2_eccentric.py` (`eccentric_phase_correction`, `eccentric_amplitude_factor`, `forward`) | `src/jaxwave/approximants/taylorf2_eccentric.py` (same components, `evaluate`) | TaylorF2 carrier and options reused. The native `f0` fallback samples the positive frequency grid; the phase is anchored at a distinct positive explicit `f_ref`, or at `f0` when no positive `f_ref` is supplied. |
| NRTidal v1/v2 | `src/torchwave/approximants/imrphenomd_nrtidal.py` | `src/jaxwave/approximants/imrphenomd_nrtidal.py` | IMRPhenomD carrier reused. Shared published coefficient values in `src/diffgw_core/nrtidal_fits.py`; Torch reexports them through `nrtidal_published_fits.py`. V1 has phase only, v2 also amplitude and self-spin. |
| NSBH | `src/torchwave/approximants/imrphenom_nsbh.py`, `nsbh_clean_phase.py`, `nsbh_clean_amplitude.py`, `nsbh_phase_baseline.py` | `src/jaxwave/approximants/imrphenom_nsbh.py`, `nsbh_phase_baseline.py` | Uses PhenomD phase structure but a distinct PhenomC-like amplitude; do not reuse PhenomD *amplitude*. JAX embeds the NRTidalv2 expression and local amplitude helpers; Torch separates them. Default affine gauge is an inherited compatibility adapter, not a published Thompson equation. |

The source audit corrected the manifest, provenance ledger, and native headers:
the eccentric SPA phase points to Moore et al. Eq. (6.26), the NSBH amplitude
model to [Pannarale et al., PRD 92, 084050](https://doi.org/10.1103/PhysRevD.92.084050),
and the HM model to [Mezzasoma & Yunes, arXiv:2203.15934](https://arxiv.org/abs/2203.15934).
The optional eccentric amplitude factor remains a project option without an
established primary derivation. These source mappings do not establish
coefficient-level or waveform-level equivalence.

## Evidence and limitations

The foundation source inspection recorded model composition and paper
locations; by itself it does **not** confirm coefficient-by-coefficient
equivalence. The later [HM leading-carrier reference](../../math/taylorf2-hm.md)
adds an executable eight-mode comparison within its declared scope.
`docs/model_manifest.json` records one dominant-only HM LAL diagnostic; two
historical eccentric catalog cases plus two separately frozen nonspinning
eccentric holdouts; and one catalog plus one holdout for each tidal/NSBH
model. The latter declared cases passed their historical CPU FP64 gates, but
the suite did not establish independent scalar reference behavior, full domain
accuracy, or physical validity. The HM `(2,2)` external diagnostic says
nothing about the seven additional modes; the scalar eight-mode checks are a
different kind of evidence. The NSBH 22-case expansion cited by
the manifest remains external-comparison evidence, and its inherited
PhenomD/remnant calibration provenance remains open.

Remaining independent-reference blockers are concrete: (1) HM tidal terms,
Kerr and smooth cutoff options, and an exact same-model external eight-mode
waveform comparison; (2) eccentric aligned-spin cross terms and provenance for
the optional amplitude factor beyond the scoped nonspinning amplitude-off
version; (3) v1 tidal phase, v2 self-spin and whole-waveform composition,
historical D-amplitude normalization, and the high-frequency binary taper
discrepancy; (4) NSBH
disruption root, every amplitude branch, supplementary remnant coefficients
and intrinsic/affine phase split. The published validity regions must be
separated from permissive native input acceptance. This ledger makes no
whole-model verification claim.

To promote the remaining models, create three `docs/math/<model>.md` pages, each with
these same six headings, exact option/front-end inventory, composition
equation, links to versioned component pages, and distinct source,
independent-reference, backend, external-comparison and physics evidence.
Implement paper-literal scalar components under `diffgw_reference`, then
compose complete scalar polarizations with fixed conventions. Test those
components separately from native Torch/JAX, add representative holdouts and
branch-boundary/derivative cases, reconcile source pins and carrier precision,
and only then consider their catalog entries, generated tables and qualification.

## Reproduce the checks

This page's static scope can be checked without LAL or waveform fixtures:

```sh
python -m json.tool docs/model_manifest.json >/dev/null
git diff --check
rg 'TaylorF2_HM|TaylorF2_Eccentric|IMRPhenomD_NRTidal|IMRPhenomNSBH' \
  src/torchwave/waveforms.py src/jaxwave/waveforms.py
```

Maintained native checks, when the relevant Torch/JAX dependencies are
installed, include `pytest -q tests/test_taylorf2_hm_math_reference.py
tests/test_taylorf2_hm.py
tests/test_taylorf2_hm_mode_inventory.py tests/test_eccentric.py
tests/parity/test_eccentric_parity.py tests/test_nrtidal.py
tests/test_nrtidal_reconstruction.py tests/test_clean_room_nsbh.py
tests/test_nsbh_phase_mass_regression.py`. The LAL-dependent test in
`test_nrtidal.py` may skip when LAL is absent; such a skip conveys no
independent-reference evidence. Refresh any frozen external fixtures only
through their separate provenance workflow. An equation, coefficient,
branch, phase convention, or asset change requires a new component version,
its source pin and tests, plus review of the generated model documentation
once catalog registration is justified.
