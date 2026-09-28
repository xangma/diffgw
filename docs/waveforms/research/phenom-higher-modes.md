# Phenom higher modes: provisional component ledger

## Scope and status

This research ledger was founded at commit 5bc907d. The separate [published-calibration HM specification](../../math/imrphenomhm-waveform.md) now defines a bounded independent six-mode composition; its catalog claims, when supported by recorded runs, concern only that explicit option tuple. This ledger retains the broader family evidence and unresolved gates. It covers aligned-spin IMRPhenomHM and IMRPhenomXHM, their inherited quadrupole carriers, and the higher-mode carrier supplied to XPHM. It excludes precession dynamics, covered in [the companion ledger](phenom-precessing.md), and all SEOBNR models. “Pending” below means that the mathematical rule has a source and a native implementation but has not crossed the independent equation, branch, and reference gates at the end of this file. “Executable” identifies a bounded equation component only.

Primary sources: [London et al., arXiv:1708.00404v4](https://arxiv.org/pdf/1708.00404v4) (HM, Eqs. 1–9); [García-Quirós et al., arXiv:2001.10914v2](https://arxiv.org/html/2001.10914v2) (XHM, Secs. II, IV–VI and Apps. A, D, E); [Pratten et al., arXiv:2001.11412](https://arxiv.org/abs/2001.11412) (XAS). Inherited D and XAS mathematics are documented separately in [D carrier specification](../../math/imrphenomd-waveform.md) and [XAS specification](../../math/imrphenomxas.md). The observed reference outcomes below come from the repository's [provenance ledger](../../provenance_ledger.md), not from a new numerical campaign here.

## Inputs, conventions, and options

Both models take positive Fourier frequencies, component masses and aligned spins, distance, inclination, coalescence phase/time, and optional reference frequency and phasing convention. Their native Fourier-time factor is exp(-2πift_c); the complex mode, polarization, and spin-weighted-harmonic signs must be held fixed in comparisons. XHM's paper explicitly uses non-negative reconstructed amplitudes, while odd-m signed fit surfaces cross zero (Sec. III; Eqs. 7, 14, 20–21). Equal-mass equal-spin odd modes therefore need their own zero/parity checks, not a generic positive-amplitude check.

| Family | Native modes and carrier | Options and domain distinctions |
|---|---|---|
| HM | Torch/JAX support (2,2), (2,1), (3,3), (3,2), (4,4), (4,3); D is default (both [Torch](https://github.com/xangma/diffgw/blob/5bc907d/src/torchwave/approximants/imrphenom_hm.py) and [JAX](https://github.com/xangma/diffgw/blob/5bc907d/src/jaxwave/approximants/imrphenom_hm.py)). | Torch also accepts an XAS carrier or injected base_22. Those are adapters, not the London D-carrier calibration. Select modes and explicitly pin carrier, f_ref, phasing convention, phic, and tc. |
| XHM | Torch/JAX support (2,2), (2,1), (3,3), (3,2), (4,4); XAS quadrupole inherited exactly in the paper and as native carrier ([Torch](https://github.com/xangma/diffgw/blob/5bc907d/src/torchwave/approximants/imrphenom_xhm.py), [JAX](https://github.com/xangma/diffgw/blob/5bc907d/src/jaxwave/approximants/imrphenom_xhm.py)). | Both frontends accept calibration labels sovereign, torchwave_nr, native, torchwave and experimental_phase_connections (default false). These label different fit sources, so a reference case must name the label. The experimental true branch is not the default waveform. |

The v2 XHM paper numbers equations consecutively (e.g. inspiral Eqs. 25–33, intermediate Eq. 37, ringdown Eq. 42); comments in native code sometimes use section-style labels such as “Eq. (4.11)” and “Eq. (5.3)”. A final specification must pin the document version and reconcile these labels before quoting an equation as an oracle.

## Mathematics and components

The component IDs below retain their earlier research scopes. The new independent published-calibration composition has separate catalog component IDs on its canonical page. Dependencies marked “D” or “XAS” are established in their own specifications, and do not certify the new assembly.

| Provisional ID / status | Rule and paper equation | Dependency and native mapping | Missing independent evidence |
|---|---|---|---|
| HM-FMAP / executable (`hm.fmap.eq5.v1`, `hm.fmap.phase.eq8.v1`, `hm.fmap.selection.london2018.v1`) | [Component specification](../../math/imrphenomhm-frequency-map.md): London Eq. 5 maps each mode's frequency to the 22 carrier; the D-carrier phase uses Eq. 8 above ringdown. The paper-printed outer-map cutoffs are an opt-in public D-carrier setting; the default retains the prior runtime pair. | [Scalar reference](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/imrphenomhm_frequency_map.py), [Torch frequency maps](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/imrphenom_hm.py), [JAX maps](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/imrphenom_hm.py), [maintained math-reference tests](https://github.com/xangma/diffgw/blob/main/tests/test_imrphenomhm_frequency_map_math_reference.py). | The public selection gate covers all six D-carrier modes at two remnant-producing mass/spin inputs for both transition pairs. It does not qualify QNM fits, phase offsets, or full higher-mode assembly. Synthetic QNM inputs separately test all six maps, including (4,3), but do not qualify those upstream inputs. |
| HM-QNM-N61 / bounded executable (`hm.qnm.n61.offgrid-six-mode/v1`) | [Off-grid component specification](../../math/imrphenomhm-qnm-n61-offgrid.md): London--Fauchon-Jones Eqs. (19)--(21) polynomial families for the named clean-room N61 refit, checked against Cook--Zalutskiy Kerr eigenproblem roots. | Frozen independent roots for all six fundamental modes at oriented spins `-0.71`, `0.37`, `0.79`; Torch/JAX N61 evaluators and public total-mass adapter checked on CPU FP64. | The hypothesized 61-node training grid is not historical author data. Near-extremal/full-domain error, remnant-provider accuracy, and full HM assembly remain open. |
| HM-AMP / partial: `hm.amp.beta33.leading.eq4.v1`, `hm.amp.beta44.leading.eq4.v1`, `hm.amp.beta21.1p5pn.eq4.v1`, `hm.amp.beta32.1p5pn.eq4.v1`, and `hm.amp.beta43.1p5pn.eq4.v1` executable | [Leading (3,3)](../../math/imrphenomhm-amplitude-ratio.md), [leading (4,4)](../../math/imrphenomhm-amplitude-ratio-44.md), [(2,1) through 1.5PN](../../math/imrphenomhm-amplitude-ratio-21.md), [(3,2) through 1.5PN](../../math/imrphenomhm-amplitude-ratio-32.md), and [(4,3) through 1.5PN](../../math/imrphenomhm-amplitude-ratio-43.md): London Eqs. 3–4 scale the carrier with Mishra Eqs. 11 and 12a/b/c/d/f/g. | Independent SymPy/mpmath references and public D-carrier Torch/JAX CPU-FP64 value and one-sided slope gates for both outer-map settings. The (2,1), (3,2), and (4,3) source gates select `mishra_eq12b`, `mishra_eq12d`, and `mishra_eq12g` respectively; defaults remain binary-compatible. | London's declared truncation is leading frequency for ℓ=|m| and 1.5PN for |m|=ℓ−1; higher PN terms are a different variant, not a missing term in this one. Negative-frequency reconstruction and independent D amplitude and QNM inputs remain open. Complex intrinsic assembly now has a conditional gate below. Literal (2,1) and (4,3) Eq. 4 ratios are undefined at equal mass and equal spin. |
| `hm.phase.ringdown.eq9-c1.v1` / executable component | [Ringdown phase-shape specification](../../math/imrphenomhm-ringdown-phase.md): London Eq. (9) modifies Khan/PhenomD Eq. (14)'s arctangent, with a native D-carrier C1 attachment. | Independent 70-digit scalar reference and production Torch/JAX values, gradients, join, and 22 control with supplied alpha/QNM inputs. | Alpha fits, D/HM QNM convention and physical provenance, full phase gauge/attachment, and complete modes remain unqualified. |
| `hm.phase.public-composition.eq3-14.v1` / executable component | [Public intrinsic phase composition](../../math/imrphenomhm-phase-composition.md): London Eqs. (3), (5)–(9) compose the corrected Khan/PhenomD carrier across all three branches. | Independent scalar composition compared with public Torch/JAX unprojected phase for five non-22 modes and both outer-map settings, including one-sided slopes at the two HM joins and mapped D attachment. | Conditional on supplied D phase, alpha fits and HM/D QNMs; amplitude sign/zeros, nonzero phase/time gauges, complete modes, binary and physical accuracy remain separate gates. The stable `eq3-14` ID refers to PhenomD Eq. (14). |
| `hm.intrinsic.complex-assembly.eq3.v1` / conditional executable component | [Six-mode complex intrinsic specification](../../math/imrphenomhm-intrinsic-modes.md): London Eqs. (2)–(9) combine the mapped D amplitude, restricted PN ratios and corrected phase. The public adapter fixes the native Fourier sign, odd-mode sign, and orbital/time gauges. | Scalar SymPy/mpmath assembly and public Torch/JAX CPU-FP64 complex-mode values for all six modes, both outer-map settings, and samples on both sides of joins; [original run receipt](../../validation/imrphenomhm_intrinsic_modes_20260927.md) and [finite-reference extension](../../validation/imrphenomhm_sxs_conventions_20260927.md). | Conditional on frontend-supplied D amplitude/phase, phase-fit coefficients and HM/D QNMs/slope. The test uses one asymmetric mass/spin point, `f_ref=None` and `40 Hz`, and no independent physical-accuracy oracle; fit provenance, other gauges and physical negative-frequency reconstruction remain open. |
| `hm.conventions.sxs-bridge.v1` / executable convention component | [HM–SXS convention specification](../../math/imrphenomhm-sxs-conventions.md): London mode expansion, García-Quirós Fourier pair/rotation and SXS current strain sign. | SymPy identities and direct sampled chirp integrals test frequency-reversing conjugation, native positive carrier, time/orbital gauges and a doubled relative invariant; the [receipt](../../validation/imrphenomhm_sxs_conventions_20260927.md) includes the finite-`f_ref` public-mode check. | No NR-derived phase correction and no change to the red SXS physical holdouts. The quarter-cycle phase in XHM's stationary-phase discussion is not a proven HM–SXS conversion. |
| HM-PROJ / executable component (`hm.projection.london-eq1.v1`) | [Six-mode projection specification](../../math/imrphenomhm-projection.md): London Eq. (1) spin -2 expansion with García-Quirós v2 Eqs. (2), (4)–(9) fixing the positive-frequency pair, plus native carrier/azimuth adapter and common time shift. | Independent Wigner/SymPy/mpmath coefficients and public Torch/JAX complex per-mode and six-mode plus/cross at pole, edge, generic inclinations, nonzero azimuth/phase/time, and odd-mode symmetry. | Intrinsic carrier and QNM inputs, negative-frequency reconstruction, and full HM waveform remain separate gates. A distinct [six-mode LAL binary holdout](../../validation/imrphenomhm_lal_six_modes_20260927.md) checks selected aligned-spin intrinsic carriers. |
| XHM-INS-AMP / pending | XHM v2 Eqs. 25–28 and App. E PN amplitudes: mode-dependent MECO/ISCO cutoffs, H_lm=|A_lm|/A_22^0 and three pseudo-PN collocation terms. Sec. IV.1.2 specifies mode-specific vetoes. | XAS amplitude and XHM PN/fit surfaces in [Torch](https://github.com/xangma/diffgw/blob/5bc907d/src/torchwave/approximants/imrphenom_xhm.py), [JAX](https://github.com/xangma/diffgw/blob/5bc907d/src/jaxwave/approximants/imrphenom_xhm.py). | Full published veto ordering, absolute-vs-normalized amplitude convention, author fit provenance and boundary cases. The native absolute-amplitude Table VI-type veto is deliberately inactive because normalization is unresolved. |
| `xhm.phase.inspiral.carrier-rescaling.v1` / executable | XHM v2 Eq. 29 and App. D Eqs. 61–63 rescale the XAS 22 phase and its power/log coefficients. | [Independent component specification](../../math/imrphenomxhm-inspiral-carrier.md), scalar SymPy reference, and clean Torch/JAX helpers used by both XHM modules. | Exact algebra, frontend values, Torch gradient, and JAX JIT are maintained. This slice excludes every mode correction and attachment term. |
| XHM-INS-PHASE / conditional | XHM v2 Eqs. 30–34 add the PN complex-amplitude argument or high-q fit, the calibrated linear attachment term, and the phase constant to the executable carrier slice above. | XAS phase; both XHM modules. Default native carrier-based phase differs from the unavailable experimental C1 connection. | Pin phase/time gauge and independently derive the correction and connection before treating the complete inspiral phase as paper-equivalent. |
| XHM-INTER / pending | XHM v2 Eqs. 35–40 and Sec. V: per-mode ringdown starts, inverse fifth-order intermediate amplitude (Eq. 37), EMR pre-intermediate branch and phase-derivative collocation. | XAS cutoffs; XHM fitted surfaces in both frontends. | Fit inputs and branch/veto transcription for all modes; C1 values and derivatives at every attachment; EMR q>70 and high-spin cases. |
| `xhm.phase.intermediate.nonmix-collocation.v1` / executable component | [Component specification](../../math/imrphenomxhm-intermediate-phase.md): XHM v2 Eqs. 38–39 and Table 7 fix the five-point phase-derivative collocation for 21/33/44. | Independent mpmath reference and Torch/JAX helpers used by production. | Supplied targets only; the fitted target surfaces, attachment, 32 mode, and full intrinsic modes remain unqualified. |
| `xhm.phase.32.intermediate-ringdown-join.v1` / executable component | [Component specification](../../math/imrphenomxhm-transition-mixing.md): XHM v2 Eqs. 38 and 40 and Table 7 fix the four (3,2) nodes and six-condition intermediate-phase solve. | Independent 70-digit scalar reference; production Torch/JAX nodes, coefficients, phase, derivatives, batching, and target gradient checked on synthetic inputs. | The fitted target surfaces, spherical mixed ringdown calibration, attachment frequencies, and full intrinsic mode remain unqualified. |
| XHM-RD / pending | XHM v2 Eqs. 41–end of Sec. VI: QNM-centered amplitude (Eq. 42), rescaled 22 phases for 21/33/44, and 32 spheroidal↔spherical mixing (App. A). | XAS remnant/QNMs plus [native XHM fits](https://github.com/xangma/diffgw/blob/5bc907d/src/torchwave/approximants/xhm_native_fits.py) and [JAX counterpart](https://github.com/xangma/diffgw/blob/5bc907d/src/jaxwave/approximants/xhm_native_fits.py). | Independent complex 32 mixing, Kerr QNM scaling and fit provenance; whole-mode amplitude+phase validation. |
| `xhm.amplitude.ringdown.ansatz.v1` / executable component | [Component specification](../../math/imrphenomxhm-ringdown-amplitude.md): XHM v2 Eq. 42 fixes the QNM-centered ringdown amplitude shape for supplied parameters. | Independent 70-digit scalar reference and production Torch/JAX batched values and gradients on safe positive-frequency fixtures. | Fitted parameters, QNM frequencies, physical normalization, interval selection, (3,2) basis mixing, and full modes remain unqualified. |
| `xhm.phase.ringdown.nonmix-c1.v1` / executable component | [Component specification](../../math/imrphenomxhm-nonmix-ringdown-phase.md): XHM v2 Eqs. 43–44 and the paragraph after Eq. 45 fix the 21/33/44 phase shape and C1 attachment for supplied boundary data. | Independent 70-digit scalar reference and production Torch/JAX shifts, batched phase, frequency/parameter gradients, and join checks. | Eq. 45 coefficient rescaling, fitted surfaces, QNM provenance, phase gauge, (3,2) mode, and complete modes remain unqualified. |
| `xhm.positive-frequency.mode-pair-projection.v1` / executable component | [Projection specification](../../math/imrphenomxhm-projection.md): XHM v2 Sec. II Eqs. 2, 4–9 fix the negative-`m` positive-frequency carrier and both polarization signs; include common time/orbital phase. | Independent Wigner-`d` scalar reference and five-mode Torch/JAX math-reference gate. | Intrinsic carrier amplitudes/phases, LAL complex comparisons, and XPHM precessing rotation remain separate full-model gates. |

The XHM native ringdown fit route was built from SXS NR data and native Kerr QNMs, as recorded in the [provenance ledger](../../provenance_ledger.md). It is a distinct calibration route from a literal paper coefficient transcription. XAS itself has an opt-in check of the author-published fit asset (SHA256 94b9334c19f3a7909864891ea278ab9a411e4f33b7b7e17cc01d927056d9ac52) on sampled inputs; it does not establish XHM fit provenance or NR accuracy.

## Implementation mapping

The public entry points are [Torch dispatcher](https://github.com/xangma/diffgw/blob/5bc907d/src/torchwave/waveforms.py) and [JAX dispatcher](https://github.com/xangma/diffgw/blob/5bc907d/src/jaxwave/waveforms.py). Core carriers are [Torch D](https://github.com/xangma/diffgw/blob/5bc907d/src/torchwave/approximants/imrphenomd.py), [JAX D](https://github.com/xangma/diffgw/blob/5bc907d/src/jaxwave/approximants/imrphenomd.py), [Torch XAS](https://github.com/xangma/diffgw/blob/5bc907d/src/torchwave/approximants/imrphenom_xas.py), and [JAX XAS](https://github.com/xangma/diffgw/blob/5bc907d/src/jaxwave/approximants/imrphenom_xas.py). HM's D inheritance is direct; its optional XAS carrier is a code-level compositional extension. XHM has a paper-level XAS 22 dependency. XPHM imports XHM, so every unsettled XHM mode/calibration/phase branch propagates to XPHM.

The [HM reconstruction tests](https://github.com/xangma/diffgw/blob/5bc907d/tests/test_phenomhm_reconstruction.py) exercise frequency mapping, joins, parity, six finite modes, gradients, and batches. The [XHM fit tests](https://github.com/xangma/diffgw/blob/5bc907d/tests/test_xhm_native_fits.py) exercise selected native ringdown surfaces and restricted phase quality gates. These are useful implementation tests; they do not independently reconstruct every paper fit.

## Evidence and limitations

The published composition requires `calibration="published"` and the three
named Mishra PN options, and independently evaluates either transition setting.
The [reviewed composition run](../../validation/imrphenomhm_published_composition_20260927.md)
passes its declared values, mapped joins, projection and selected smooth
derivative checks in both frontends; the catalog records that bounded result.
Selecting calibration alone leaves `binary_compatible` PN defaults in place.
Its mathematical implementation checks cannot inherit the default variant's
binary or NR results. All historical outcomes below, including red gates,
remain unchanged. The exact published option with `london_2018` mapping now
has its own [prospectively frozen two-case NR result](../../validation/imrphenomhm_published_nr_receipt_20260927.md).
SXS:BBH:2374 and 2163 pass their NR resolution screens and first model view,
but each fails the second view's match and norm gates in both frontends.
The matches are 0.971608 and 0.972273, and norm ratios 0.750351 and 1.223703.
This option therefore remains physically unqualified under those gates.

A [controlled option and phase-gauge diagnostic](../../validation/imrphenomhm_published_options_20260927.md)
finds a finite-reference covariance defect if `f_ref` is intended only to set
the orbital phase origin. The predicted extra analytic phase is reproduced by
both frontends and the independent scalar carrier. Diagnostic factor removal
restores orbital covariance but leaves NR disagreement. Runtime mapping helps
the spinning case; none of the tested options closes both cases. These seen-case
probes leave all frozen outcomes unchanged.

The historical [frozen catalog and holdout](../../provenance_ledger.md#3-reference-evidence-by-model) each contain one passing HM case at declared complex relative L2 ≤ 10^-6 and mismatch ≤ 10^-8. That is case-specific and does not establish a domain bound. XHM has 0/1 passing full-waveform cases in each set at those strict gates. Its separate native phase tests use restricted ringdown windows, reference amplitudes and affine phase alignment, so they cannot be cited as whole-waveform accuracy. The paper's own NR quality control is evidence for the published model, not automatically for either native implementation or the alternate native-fit route.

The [maintained pinned-binary six-mode holdout](../../validation/imrphenomhm_lal_six_modes_20260927.md) compares each native HM/D complex carrier against LALSuite 7.26.15 on three independently selected aligned-spin systems, with a fixed Fourier/orbital-phase adapter and no per-case alignment. Both frontends pass the declared integrated and supported-bin residual gates for all six modes. This is sampled implementation agreement with the same-model binary; it does not replace independent equation/fitted-input provenance or NR accuracy evidence.

The separate [pinned-binary plus/cross holdout](../../validation/imrphenomhm_lal_polarizations_20260927.md) checks public Torch/JAX HM/D polarizations at three additional aligned-spin systems with generic inclinations, azimuths, nonzero orbital phases, and time shifts. Both frontends pass the frozen complex full-band and strong-bin residual bounds for plus and cross without per-case alignment. This adds sampled projection-plus-carrier binary evidence; it is not a full-domain or NR accuracy claim.

Prospective full-NR physical comparisons now pass one nearly nonspinning
[SXS:BBH:3144](../../validation/imrphenomhm_sxs_3144_receipt_20260927.md)
case and one aligned-spin
[SXS:BBH:2158](../../validation/imrphenomhm_sxs_2158_receipt_20260927.md)
case at the precommitted two-view resolution, match, and norm gates. The
earlier [2265](../../validation/imrphenomhm_sxs_2265_receipt_20260927.md),
[3631](../../validation/imrphenomhm_sxs_3631_receipt_20260927.md),
[3619](../../validation/imrphenomhm_sxs_3619_projection_receipt_20260927.md),
and [3630](../../validation/imrphenomhm_sxs_3630_projection_receipt_20260927.md)
red results remain red. The two green cases add bounded physical evidence;
they do not qualify the whole aligned-spin domain, every view, or the
inherited fit/QNM provenance.

The [2158 closeout diagnostic](../../validation/imrphenomhm_sxs_2158_receipt_20260927.md#post-result-diagnostic-and-case-closeout)
reproduces the weaker-view match with native HM (`0.98649901`) and the
pinned LAL binary (`0.98649896`) under the same fixed alignment and
reference projection. Its frequency-resolved complex residual and
six-pair NR comparison point away from a frontend discrepancy or missing
modes alone. The case is closed with bounded passing evidence; the
diagnostic does not establish a unique physical cause, repair earlier
failures, or promote HM to whole-family qualification.

The maintained mode/regime diagnostic now localizes those two XHM failures without changing their gates. The 22 mode agrees with the pinned binary to numerical precision after one constant phase convention. Across 21, 32, 33 and 44, affine-aligned inspiral phase residuals are at most 1.33e-4 rad, while intermediate phase residuals span 6.24–28.80 rad. Intermediate amplitude relative L2 spans 0.77–7.04; the largest inspiral-amplitude residual is 0.48 for the catalog 32 mode. Ringdown residuals and the 32 mixing path remain mode-dependent, and the close full-polarization/projection metrics in the manifest only show that projection is not the dominant amplifier in these two cases. This black-box decomposition is diagnostic evidence, not a calibration fit or a complete-waveform verification claim.

Paper/source discrepancies to resolve: HM offers a non-paper XAS carrier, and its default `phenomd_runtime` assembly reverses London's printed amplitude and phase transition frequencies, while the explicit `london_2018` option follows them; XHM's default and experimental phase joins differ; some XHM absolute amplitude vetoes remain disabled pending normalization; XHM code comments use equation labels that differ from the pinned v2 paper; fit-source labels are not interchangeable. None of these should be hidden behind a single unqualified model name.

## Reproduce the checks

Read-only source anchors: the two papers above, the four native HM/XHM modules, [XHM fit source](https://github.com/xangma/diffgw/blob/5bc907d/src/torchwave/approximants/xhm_native_fits.py), [provenance](../../provenance_ledger.md), and frozen [manifest](../../model_manifest.json). Minimal local implementation checks: run the HM reconstruction and XHM fit test files named above with the repository's normal pytest environment; the LAL/NR portions may require optional dependencies or data. Run `tests/test_imrphenomhm_lal_modes_binary.py` and `tests/test_imrphenomhm_lal_polarizations_binary.py` with pinned LALSuite and `DIFFGW_REQUIRE_LAL_BINARY=1` for the maintained six-mode and plus/cross external gates. Run `tests/test_xhm_inspiral_carrier_phase.py` for the independently referenced XHM carrier slice and `scripts/diagnose_xhm_frozen_components.py` for the optional XHM LAL-binary mode/regime decomposition. The diagnostic records exact package versions and does not use oracle output to modify production coefficients.

Qualification must name its claim category and **each option tuple and mode**.
A bounded mathematical implementation claim requires pinned source equations
and fits, declared numerical assets, fixed units and phase/Fourier conventions,
and independent scalar-to-Torch/JAX comparisons with recorded values,
derivatives, branches and symmetry scope. The
[published-calibration HM specification](../../math/imrphenomhm-waveform.md)
provides that separate executable composition and maintained gates; catalog
claims require reviewed passing evidence. It does not cover inherited
calibration or binary-compatible PN defaults.

Same-model binary agreement requires separate declared catalog/holdout mode
and polarization comparisons with fixed metrics and tolerances. Physical
accuracy requires separate SXS/NR comparisons and uncertainty control. Neither
category follows from a passing math reference. XHM's unresolved fit surfaces,
vetoes and 32 mixing, and all red NR gates above, remain open. Pin HM/D and
XHM/XAS dependencies and record commands, versions, fixture hashes and observed
outcomes before extending any claim.

The [orbital reference-phase extension](../../validation/imrphenomhm_orbital_reference_20260928.md)
adds an explicit project convention and independent covariance checks while
preserving legacy behavior. The [mapping diagnostic](../../validation/imrphenomhm_orbital_mapping_20260928.md)
separates phase and amplitude effects on the already examined 2374/2163 data.
Improved scores after phase-map changes do not establish a source correction
or prospective physical qualification; the original gates remain unchanged.

The [modal-phase diagnostic](../../validation/imrphenomhm_modal_phase_20260928.md)
finds substantial 33/44 relative-phase offsets and mapping-dependent phase
evolution on those same fixtures, well above their measured resolution
differences in the transition bands. The 22 carrier remains a strong control.
A source-derived convention check is needed before interpreting a constant
offset as a production defect; changing the mapping does not remove it.

The [SPA convention reference and raw-TD check](../../validation/imrphenomhm_spa_conventions_20260928.md)
now derives that branch conversion from published coefficients, the strain
basis and the Gaussian stationary-phase limit. Applying its fixed factor in
a scratch comparison reduces the transition-band offsets substantially, but
leaves later phase drift and magnitude errors. Because the factor decomposes
into a common carrier phase and orbital rotation, signed-mode projection and
gauge transport must be established before production use. This reference-only
slice leaves the catalog and physical qualification gates unchanged.
