# TaylorF2_Eccentric nonspinning low-e mathematical specification

<!-- BEGIN GENERATED WAVEFORM DEFINITION -->

| Definition field | Catalog value |
| --- | --- |
| Model ID | `TaylorF2_Eccentric` |
| Reference composition | `taylorf2-eccentric-nonspinning-e2-3pn/v1` |
| Scope | Nonspinning comparable-mass (q <= 2), 0 < e0 <= 0.1, positive-frequency inspiral: restricted quadrupole amplitude and 3PN order-e0² eccentric phase at explicit f0_ecc and f_ref, with amplitude correction off. |
| Domain and modes | `frequency`; `(2, 2)` |
| Native backends | `torch`, `jax` |
| Scoped verification claims | 1 |

| Option | Default | Native choices | Reference-only choices |
| --- | --- | --- | --- |
| `phasing_convention` | `coalescence` | `coalescence`, `reference_frequency` | None |
| `ecc_order` | `6` | `0`, `2`, `3`, `4`, `5`, `6` | None |
| `include_amplitude_correction` | `false` | `false`, `true` | None |
| `f_isco_cutoff` | `true` | `false`, `true` | None |

<!-- END GENERATED WAVEFORM DEFINITION -->

## Scope and status

Specification `taylorf2-eccentric-nonspinning-e2-3pn/v1` composes the unchanged
[nonspinning TaylorF2 3.5PN carrier](taylorf2.md) with the
[paper-derived 3PN eccentric phase](taylorf2-eccentric-phase.md). It covers the
positive-frequency, dominant-harmonic, restricted-amplitude inspiral with
zero spins, comparable masses $1\le q=m_1/m_2\le2$, and
$0<e_0\le0.1$. The eccentricity is defined at an explicit $f_{0,\rm ecc}$.
Both native backends and the scalar reference use an explicit, positive
$f_{\rm ref}$ for the project reference-frequency phase convention.

The native API accepts more parameter choices than this versioned slice.
Optional eccentric amplitude, aligned-spin terms and their possible
spin-eccentric couplings, other phase origins, arbitrary mass ratios, larger
eccentricities, and cutoff-boundary behavior remain provisional. This is a
bounded mathematical and implementation qualification, not a uniform physical
accuracy statement.

## Inputs, conventions, and options

Masses are positive solar masses ordered $m_1\ge m_2$, distance is positive
Mpc, $0\le\iota\le\pi$, and $t_c,\phi_c$ are finite. The grid frequencies,
$f_{0,\rm ecc}$, and $f_{\rm ref}$ are positive Hz. The sampled band stays
below Schwarzschild $f_{\rm ISCO}=[6^{3/2}\pi M]^{-1}$, with
$M=G(m_1+m_2)/c^3$. Thus the native default `f_isco_cutoff=True` has no
effect on the qualified samples. The scalar reference is unwindowed; no
assertion is made at the hard cutoff or after it.

Select `chi1=chi2=0`, no tides, `ecc_order=6`,
`include_amplitude_correction=False`, and
`phasing_convention="reference_frequency"`. Supply `e0`, `f0_ecc`, and
`f_ref` explicitly. `ecc_order=6` selects the relative **3PN** eccentric
phase at order $e_0^2$; it does not select order $e_0^6$. The native API's
implicit reference-frequency priority and other PN truncations are described
in the [phase component specification](taylorf2-eccentric-phase.md), outside
this composition claim. The strict external cases use CPU float64.

## Mathematics and components

Let $v=(\pi Mf)^{1/3}$, $v_0=v(f_{0,\rm ecc})$, and
$\eta=m_1m_2/(m_1+m_2)^2$. The circular intrinsic phase $I_{3.5\rm PN}$
and restricted quadrupole amplitude $A_0$ are exactly the versioned
[TaylorF2 base expressions](taylorf2.md). The eccentric term from
[Moore et al., Eq. (6.26)](https://arxiv.org/pdf/1605.00304v2) is

$$
E(f;f_{0,\rm ecc})=-\frac{2355}{1462}\frac{3e_0^2}{128\eta v^5}
\left(\frac{v_0}{v}\right)^{19/3}P_{3\rm PN}(v,v_0,\eta).
$$

The complete $P_{3\rm PN}$ polynomial, including distinct logarithms in
$v$ and $v_0$, is transcribed and tested in the
[phase component specification](taylorf2-eccentric-phase.md). The composed
project phase and polarizations are

$$
\Psi(f)=2\pi ft_c-2\phi_c+\frac{3\pi}{4}
+I_{3.5\rm PN}(f)-I_{3.5\rm PN}(f_{\rm ref})
+E(f;f_{0,\rm ecc})-E(f_{\rm ref};f_{0,\rm ecc}),
$$

$$
h_+(f)=\tfrac12(1+\cos^2\iota)A_0(f)e^{-i\Psi(f)},\qquad
h_\times(f)=-i\cos\iota\,A_0(f)e^{-i\Psi(f)}.
$$

The two $f_{\rm ref}$ subtractions and $3\pi/4$ offset are DiffGW phase
conventions, not terms in Moore's paper equation. The amplitude contains no
eccentric factor in this slice. Reused `tf2.ns35.*.v1` components retain the
base version; `tf2.ecc.paper_phase.v1`, `tf2.ecc.reference_phase.v1`, and
`tf2.ecc.polarizations.v1` identify the extension and composition.

## Implementation mapping

[`src/diffgw_reference/taylorf2_eccentric.py`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_eccentric.py)
builds `waveform_expr()` from the independent base `waveform_expr()` and its
own Moore polynomial, then evaluates it at high precision through
`evaluate_waveform()`. It does not import either tensor frontend or LAL.
The native [`Torch`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2_eccentric.py)
and [`JAX`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2_eccentric.py) classes
add the eccentric phase to their existing TaylorF2 carrier; their
`definition_id` metadata binds this scoped catalog entry. No production
waveform coefficients or numerical paths changed for this promotion.

## Evidence and limitations

The [phase receipt](../validation/taylorf2_eccentric_phase_20260927.md)
records the direct Eq. (6.26) transcription and strict Torch/JAX CPU float64
component checks. The [frozen two-case holdout receipt](../validation/taylorf2_eccentric_nonspinning_holdout_20260927.json)
records nonzero-eccentricity, off-catalog direct LAL `TaylorF2Ecc` comparisons
with different mass ratios, $e_0$, $f_{0,\rm ecc}$, and $f_{\rm ref}$. Both
backend polarizations and six detector projections passed the unchanged
strict complex-$L_2$ and mismatch gates. The test uses the existing PyCBC
optimized match and named PSDs; no phase or time shift was fitted to the
complex-$L_2$ results. These two points do not establish an error envelope
over the stated domain or the accuracy of the underlying approximation.

The native aligned-spin/eccentric combination is measured in an older frozen
catalog row but has no independent paper-backed cross-term derivation here.
The optional native amplitude multiplier also lacks primary-source
provenance. Neither is included in this model version.

## Reproduce the checks

With LALSimulation, PyCBC, Torch, JAX, SymPy and mpmath installed:

```sh
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src:tests \
  python -m pytest -q tests/test_taylorf2_eccentric_holdout.py \
  tests/test_taylorf2_eccentric_math_reference.py \
  tests/test_waveform_definitions.py tests/test_parity_harness_controls.py \
  --reference-device=cpu
```

`python scripts/generate_waveform_library_table.py --check` checks the
catalog-generated header and waveform-library row. Changing the base phase,
Moore polynomial, eccentricity reference, or project phase origin requires a
new composition version and renewed scalar and external checks.
