# IMRPhenomD full intrinsic composition

<!-- BEGIN GENERATED WAVEFORM DEFINITION -->

| Definition field | Catalog value |
| --- | --- |
| Model ID | `IMRPhenomD` |
| Reference composition | `imrphenomd-intrinsic-composition/v1` |
| Scope | Aligned-spin (2,2) IMR composition with declared fit rows and QNM data. |
| Domain and modes | `frequency`; `(2, 2)` |
| Native backends | `torch`, `jax` |
| Scoped verification claims | 0 |

| Option | Default | Native choices | Reference-only choices |
| --- | --- | --- | --- |
| `calibration` | `inherited` | `inherited`, `published` | None |
| `phasing_convention` | `coalescence` | `coalescence`, `reference_frequency` | None |

<!-- END GENERATED WAVEFORM DEFINITION -->

## Scope and status

Generate the dominant-mode aligned-spin `IMRPhenomD` waveform through the
[frequency-domain API](../guides/waveform-generation.md) with either backend.
The higher-mode and NRTidal variants add physics beyond this specification.

The executable
[`imrphenomd_intrinsic.py`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/imrphenomd_intrinsic.py)
connects the unchanged [remnant and QNM specification](imrphenomd-remnant-qnm.md)
to the unchanged [complex-polarization specification](imrphenomd-waveform.md).
This extension tests how the default waveform responds to mass ratio and both
aligned spins when the remnant, QNM frequencies, amplitude, phase, joins, and
peak time gauge all move together. It imports no production waveform, fit, or
QNM-data module.

## Inputs, conventions, and options

The public inputs are component masses in solar masses, aligned spins,
distance in Mpc, inclination and orbital phase in radians, time shift in
seconds, and a frequency grid in Hz. The default phase uses the coalescence
convention; `reference_frequency` subtracts the intrinsic phase at a positive
`f_ref`. Supplying `f_ref` without an explicit `phasing_convention` selects
`reference_frequency` in both native dispatchers. An explicit convention takes
precedence over this implicit selection; `reference_frame` overrides
`phasing_convention` if both are supplied. An unsupported value raises an error.
For `reference_frequency` with no `f_ref`, the carriers use the first evaluation
frequency as the anchor. This composition needs declared fit rows and QNM knot
data. The [waveform composition](imrphenomd-waveform.md) specifies SI
normalization,
polarization signs, and the peak time gauge; the
[remnant and QNM specification](imrphenomd-remnant-qnm.md) specifies its
interpolation and high-spin continuation. The generated option table above
describes which choices the native carriers accept.

The native `calibration` option defaults to `inherited`, preserving the existing
extra fit digits. Explicit `calibration="published"` uses the printed Khan
Table V and Husa remnant constants consistently in amplitude, phase, remnant
and peak slope. QNM data and their interpolation prescription are unchanged.
The scalar reference's printed defaults remain separately transcribed; the
older comparisons below still inject inherited rows at the test boundary.
The [published-calibration HM composition](imrphenomhm-waveform.md) uses this
new native option to test a complete source-defined six-mode composition.
Published calibration does not accept a custom remnant provider.

## Mathematics and components

At fixed total mass $M$, define $q=m_1/m_2$, so
$m_1=Mq/(1+q)$ and $m_2=M/(1+q)$. The component masses and spins determine
$\eta$, $S$, and $\hat S$ in [Husa et al., Paper I,
Section III.1](https://arxiv.org/html/1508.07250v2). Paper I Eqs. (22)–(24)
then give final spin $a_f$ and final mass $M_f$. An explicitly supplied natural
cubic spline $\nu_j(a_f)$ of numerical cyclic Kerr QNM ordinates supplies the
ringdown and damping channels $j$. Paper I Eq. (17) gives their total-mass
coordinates,

$$r_j=\frac{M}{M_f}\nu_j(a_f),\qquad
\partial_p r_j=\frac{M}{M_f}\left[
\nu'_j(a_f)\,\partial_p a_f-
\nu_j(a_f)\frac{\partial_p M_f}{M_f}\right],$$

for $p\in\{q,\chi_1,\chi_2\}$ at fixed $M$. The reference obtains
$\partial_p M_f$ and $\partial_p a_f$ from the SymPy expressions for the
published remnant fits, and $\nu'_j$ from the independently constructed natural
spline. The QNM knots and ordinates are numerical inputs, not equations
published in either IMRPhenomD paper.

For the endpoint value check, the scalar reference caps the fitted spin at
$|a_f|=0.999$ and switches above $|a_f|=0.994$ to an explicitly supplied
degree-five Kerr spline. It evaluates that spline with SciPy in the transformed
coordinate $x=-\log(1-|a_f|)$, converts its angular ordinates to cyclic units
by $2\pi$, and then applies the same published total-mass scaling. The spline
coefficients are generated numerical inputs with the
[separate QNM provenance](../provenance_ledger.md#212-clean-room-kerr-quasinormal-mode-solvers-generation-torchwaveqnm),
not waveform fit coefficients or a new paper equation. This checks how the
whole waveform consumes those inputs; the independent-method QNM comparisons
remain in the [remnant/QNM audit](imrphenomd-remnant-qnm.md).

The pair $(r_{\rm RD},r_{\rm damp})$ enters the amplitude and phase expressions
of [Khan et al., Paper II, Eqs. (5)–(7), (17)–(18), and
(36)–(39)](https://arxiv.org/html/1508.07253v2). The existing scalar reference
then evaluates the positive-frequency complex polarizations with DiffGW's
documented SI normalization and phase conventions. Consequently a change in
$p$ propagates through $\eta$, $\chi_{\rm PN}$, the fitted coefficients, the
QNM-driven amplitude peak, collocation points, phase joins, and the peak phase
gauge. At fixed physical frequency, $q$ leaves $M_s f$ unchanged, whereas a
spin change leaves both $M_s f$ and the physical amplitude prefactor unchanged.

The full complex-polarization derivative is an **independent numerical slope**
of scalar reference evaluations, not a symbolic proof of the entire piecewise
waveform. For each polarization $h$ and parameter $p$, it uses

$$\partial_p h\simeq
\frac{h(p-2\epsilon)-8h(p-\epsilon)+8h(p+\epsilon)-h(p+2\epsilon)}
{12\epsilon}.$$

The intrinsic derivative comparison covers the rounded natural cubic for
$|a_f|<0.994$, both signed generated Kerr splines for $0.994<|a_f|<0.999$,
and an interior physical-spin example on the active $|a_f|=0.999$ cap.
The cap example differentiates the waveform as masses and spins vary while
the capped QNM spin stays fixed; final mass, fitted coefficients, joins, and
phase still respond. It does not assign a derivative at the cap transition.
The separate [remnant/QNM endpoint checks](imrphenomd-remnant-qnm.md) audit
the QNM data itself. The natural cubic is $C^2$ at its knots, while the
generated spline is piecewise degree five. A five-point stencil crossing a
knot need not retain its fourth-order smooth-function error law. The reference
therefore requires the center and all four samples to lie inside one open QNM
knot interval and on the same interpolation, cap, amplitude, and phase branches,
below the hard cutoff. An explicit crossing step raises an error. The default
step starts at $10^{-4}\max(1,|p|)$ for the rounded cubic and
$10^{-7}\max(1,|p|)$ for generated Kerr data, and halves up to ten times to
find an interior stencil; it raises if none is found. A center on or within
floating-point resolution of a knot or switch, or at a physical spin endpoint,
is rejected. Tests check the guard and stencil convergence. They compare
derivatives on each side of moving joins without assigning a derivative at a
join. Opt-in tapers remain outside this qualification.

## Implementation mapping

| Component ID | Evidence | Reference function | Torch/JAX check |
| --- | --- | --- | --- |
| `phenomd.intrinsic.remnant_qnm.v1` | Paper I Eqs. (17), (22)–(24); declared rounded and generated QNM data | `remnant_qnm`, `qnm_tangent` | Default and high-spin remnant/QNM values; interior analytic mass-ratio/spin tangents |
| `phenomd.intrinsic.polarizations.v1` | Paper II waveform equations, prior composed spec, and declared high-spin QNM data | `compose` | Default-remnant and high-spin complex plus/cross through both joins and both phase conventions |
| `phenomd.intrinsic.derivatives.v1` | Chain rule and fourth-order central stencil of the sourced composition | `polarization_derivative` | Full-chain $q$, $\chi_1$, $\chi_2$ complex Jacobians on rounded, signed generated, and active-cap branches against both frontends |

## Evidence and limitations

The default remnant coefficients use **only the printed Paper I digits** and
the default waveform coefficients use **only the printed Paper II digits**.
Frontend tests pass additional inherited digits and the QNM data explicitly
at the test boundary. Agreement under those inputs establishes implementation
consistency, not independent provenance or licensing of the extra digits or
physical accuracy. See the [remnant coefficient audit](imrphenomd-remnant-qnm.md)
for the disputed $\eta S^2$ term. The principal case is
$(m_1,m_2,\chi_1,\chi_2)=(45,15,0.45,-0.25)$, with $q=3$, interior final
spin, $u=M_sf=0.012,0.028,0.080$, and a reference-frequency example.
Additional derivative checks use equal masses with unequal spins,
$(30,30,0.35,-0.2)$, and negative spins, $(40,20,-0.6,-0.3)$, at $u=0.028$.
The moving-join checks sample $u=f_{\rm peak}\pm 2\times10^{-4}$ and
$u=f_{\rm RD}/2\pm 2\times10^{-4}$. These are sampled FP64 checks, not a
uniform error bound over the parameter space.

The complete high-spin comparison uses $(150,1,0.999,0.999)$ just above the
positive continuation threshold, $(1000,1,1,1)$ with fitted spin above the
$0.999$ cap, and $(2000,1,-1,-1)$ on the negative-spin continuation. For each
case it compares $M_f$, $a_f$, both QNM channels, and unaligned complex
polarizations and phase on both sides of all four moving joins, at the hard
cutoff, and under both phase conventions. These extreme mass ratios are
numerical branch probes, not physical-accuracy claims. The check shares the
generated Kerr spline coefficients with production as declared data; its
SciPy spline evaluation and symbolic waveform composition are independent of
the Torch/JAX implementation.

The high-spin derivative cases keep both component spins strictly inside their
physical domain: $(150,1,0.999,0.999)$ on the positive generated branch,
$(2000,1,-0.999,-0.999)$ on the negative branch, and
$(1000,1,0.9998,0.9998)$ on the active positive cap. They compare the real
and imaginary parts of both complex polarizations for $q$ at fixed total mass
and for each aligned spin, at $u=0.012,0.028,0.080$ with the declared phase
conventions. These are implementation-consistency checks for extreme binary
parameters, not physical-accuracy bounds.

An additional [LALSuite binary-oracle check](../validation.md#imrphenomd-intrinsic-derivatives-against-lalsuite-23-september-2026)
compares ordinary-domain complex $q$, $\chi_1$, and $\chi_2$ derivatives with
independent fourth-order finite differences of public LAL waveform values.
It tests the complete production coefficient and QNM path without supplying
inherited fit rows to the scalar reference. This same-model comparison is
separate from the paper-to-reference and reference-to-frontend checks above.

## Reproduce the checks

Run `python -m pytest -q tests/test_imrphenomd_waveform_math_reference.py` with
the optional reference, Torch, and JAX dependencies installed. The maintained
strict math-reference gate also collects this module.
