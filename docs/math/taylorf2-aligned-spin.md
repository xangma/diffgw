# Aligned-spin TaylorF2 mathematical specification

Specification: `taylorf2-aligned-bh-spin-3.5pn/v1`.

This extension composes the unchanged
[`taylorf2-nonspinning-3.5pn/v1`](taylorf2.md) reference with separately sourced
aligned black-hole spin corrections. It retains the baseline's Newtonian
amplitude, Fourier convention, phase origins, positive-frequency domain, and
unwindowed waveform. Its spin phasing covers the conservative dynamics and
flux to infinity; horizon absorption, which starts at 2.5PN for spinning black
holes, is excluded. The old equations, component IDs, API, and fixtures remain
the baseline; a zero-spin reduction checks the composition without serving as an
independent derivation of that baseline. Its bounded
[direct source review](taylorf2.md#direct-source-review) is separate evidence.

## Parameters and composition

In addition to the baseline total mass $M$ and distance $D$ in seconds, frequency
$f$ in Hz, inclination $\iota$, time $t_c$, and orbital phase parameter $\phi_c$,
use

$$
\delta=\frac{m_1-m_2}{m_1+m_2},\qquad
\eta=\frac{1-\delta^2}{4},\qquad
\chi_s=\frac{\chi_1+\chi_2}{2},\qquad
\chi_a=\frac{\chi_1-\chi_2}{2}.
$$

Here $-1<\delta<1$ and $-1\leq\chi_i\leq1$; each spin is the signed component
along the orbital angular momentum. Both signs of $\delta$ are allowed, so
exchanging body labels maps $(\delta,\chi_1,\chi_2)$ to
$(-\delta,\chi_2,\chi_1)$. Spin-induced multipoles take their black-hole values;
general compact-object quadrupoles, tides, precession, eccentricity, amplitude
spin corrections, absorption, merger, cutoffs, and tapers are outside this scope.
These analytic bounds do not establish physical accuracy at arbitrary frequency.

Let $I_{\rm NS}(v,\eta)$ be the unchanged nonspinning intrinsic phase and
$v=(\pi M f)^{1/3}$. The extension uses

$$
I_{\rm spin}=I_{\rm NS}
 +\frac{3}{128\eta v^5}\left[
 P_{\rm SO3}v^3+P_{\rm SS4}v^4
 +P_{\rm SO5}v^5(1+3\ln v)
 +(P_{\rm SO6}+P_{\rm SS6})v^6
 +(P_{\rm SO7}+P_{\rm SS7}+P_{\rm SSS7})v^7\right].
$$

SO, SS, and SSS identify terms linear, quadratic, and cubic in spin; the number
is the power of $v$ inside the PN bracket. An explicit `terms` subset includes
only the named contributions. `SO5_log` is the derived coefficient
$3P_{\rm SO5}$, selected together with `SO5`. The kernel's default is its complete
listed term set. This explicit mathematical selection is independent of the
frontends' integer options described below.

The combined phase replaces $I$ in the baseline's two
[waveform phase conventions](taylorf2.md#amplitude-polarization-and-phase-conventions).
The logarithm scale is the same $\ln v$ convention. Changing the 2.5PN scale
changes a parameter-dependent phase constant; finite-reference subtraction
cancels that constant. Comparisons hold the declared phase convention fixed.

## Sources and transformations

The spin equations are entered independently in
[`taylorf2_aligned_spin.py`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py).
Its exact expressions import the baseline reference, never production helpers
or frontend coefficients. Component evidence pins a paper version and equation;
generated expressions and fixtures do not supply an independent source audit.

The lower-order terms use [Wade et al., arXiv:1306.3901v2](https://arxiv.org/pdf/1306.3901v2),
Eq. (4) and the accompanying definitions of $\beta$, $\sigma$, and $\gamma$.
Set the quadrupole parameters to their black-hole values and specialize all spin
products to the aligned case. In that notation the bracket corrections are
$4\beta$, $-10\sigma$, and $-\gamma$, with the 2.5PN logarithm expanded in the
declared gauge.

[Mishra et al., arXiv:1601.05588v1](https://arxiv.org/pdf/1601.05588v1),
Eqs. (6a), (6b), and (7), supply the higher-order polynomial contributions in
$\chi_s,\chi_a,\delta,\eta$. The reference retains the baseline second-harmonic
$3/(128\eta v^5)$ normalization. Mishra's Eq. (4) is not used to replace that
normalization: the first-harmonic phase convention in Eq. (1) and the printed
prefactor require care. Wade's orbital-phase prefactor becomes the baseline
factor when multiplied by the harmonic number two.

### The 3.5PN spin–spin tail

`SS7` includes the quadratic part of
[Pratten et al., arXiv:2001.11412v1, Eq. (A19)](https://arxiv.org/pdf/2001.11412v1#page=25)
**and** the additional tail contribution in its
[Eq. (A24)](https://arxiv.org/pdf/2001.11412v1#page=26). Eq. (A19) alone is
incomplete for this scope. Mishra Eq. (6b) does not supply this quadratic term.
The reference includes neither Pratten's additional 4PN terms nor its
phenomenological fits and IMR joins.

An independent balance-law calculation checks this source combination. Define
$X_{1,2}=(1\pm\delta)/2$ and $a_i=X_i\chi_i$. With black-hole multipoles,
[Bohé et al., arXiv:1501.01529v1, Eqs. (3.33), (4.14)](https://arxiv.org/pdf/1501.01529v1)
give the leading quadratic coefficients in the normalized energy and flux:

$$
e_4=-(a_1+a_2)^2,\qquad
f_4=\frac{33}{16}(a_1^2+a_2^2)+\frac{31}{8}a_1a_2.
$$

The orbital tail coefficient is $f_3=4\pi$ from
[Buonanno et al., arXiv:0907.0700v1, Eq. (3.2)](https://arxiv.org/pdf/0907.0700v1#page=4).
[Messina and Nagar, arXiv:1703.08107v4, Eq. (26)](https://arxiv.org/pdf/1703.08107v4#page=6)
gives the explicit additional flux coefficient

$$
f_7^{\rm SS}=\pi\left[\frac{65}{8}(a_1^2+a_2^2)+\frac{63}{4}a_1a_2\right].
$$

In $dt/dv=-E'(v)/\mathcal F(v)$, its normalized quadratic coefficient at
$v^7$ is $q_7=-f_7^{\rm SS}+4\pi(2f_4-3e_4)$. Integrating the second-harmonic
stationary phase gives $P_{\rm SS7}=-20q_7$, hence

$$
P_{\rm SS7}=-\pi\left[\frac{815}{2}(a_1^2+a_2^2)+785a_1a_2\right].
$$

Setting $f_7^{\rm SS}=0$ in that derivation reproduces the quadratic part of
Pratten Eq. (A19): $-\pi[570(a_1^2+a_2^2)+1100a_1a_2]$. Restoring the flux
adds $20f_7^{\rm SS}$ to the phase coefficient, exactly Eq. (A24). This source
review used the pinned printed equations and a separate symbolic reduction;
it did not infer the correction from production outputs or generated fixtures.
The backend correction therefore affects selectors `spin_order >= 8` with
calibration disabled. The public default `7`, phase default `5`, negative
selectors, and calibration profiles omit `SS7` and retain their term sets.

The exact correction table is generated from the independent expressions;
`SO5_log` multiplies $v^5\ln v$, and all coefficients exclude the common
$3/(128\eta v^5)$ prefactor.

--8<-- "docs/math/taylorf2-aligned-spin-coefficients.md"

## Frontend option contract

`frontend_terms()` in the separate traceability module translates existing
frontend options into explicit reference terms. `spin_order` is a legacy recipe,
so its integer value is not a uniform PN cutoff. The coefficient and numerical
tests check the option boundaries directly. No spin recipe is changed to make
the names look like physical orders.

<!-- BEGIN GENERATED ALIGNED SPIN PROFILES -->

| `spin_order` | Terms | With `phenom_d_calibration=True` |
| --- | --- | --- |
| 0, 1, 2 | None | None |
| 3 | `SO3` | `SO3` |
| 4 | `SO3`, `SS4` | `SO3`, `SS4` |
| 5 | `SO3`, `SS4`, `SO5`, `SO6`, `SO7` | `SO3`, `SS4`, `SO5`, `SO6`, `SO7` |
| 6, 7 | `SO3`, `SS4`, `SO5`, `SO6`, `SS6`, `SO7` | `SO3`, `SS4`, `SO5`, `SO6`, `SO7` |
| 8 or greater | `SO3`, `SS4`, `SO5`, `SO6`, `SS6`, `SO7`, `SS7`, `SSS7` | `SO3`, `SS4`, `SO5`, `SO6`, `SO7` |
| negative | `SO3`, `SS4`, `SO5`, `SO6`, `SS6`, `SO7` | `SO3`, `SS4`, `SO5`, `SO6`, `SO7` |

<!-- END GENERATED ALIGNED SPIN PROFILES -->

The public waveform default is `spin_order=7`; the low-level phase default is
`spin_order=5`. Negative values share the public default's listed terms and do
not enable the final `SS7` and `SSS7` terms. An explicit `spin_order=8` selects all
terms in this reference. `phenom_d_calibration=True` suppresses `SS6`, `SS7`, and
`SSS7` at any selector; it is available for coefficients and low-level phase,
but is not exposed by the public waveform call. This checks that compatibility
recipe, not an IMRPhenomD calibration or complete IMRPhenomD waveform.

## Component traceability

The generated map records each new component and every unchanged inherited
component version, with links to its evidence, reference expression, both
frontends, and maintained checks. For a spin correction, a coefficient tuple
position identifies the containing coefficient; the frontend does not expose
each physical spin term as a separate return value. Tests compare selected
combinations and their differences. The inherited base map remains authoritative
for its original component definitions, including its reference-only paper gauge.

<!-- BEGIN GENERATED ALIGNED SPIN TRACEABILITY -->

New components: `taylorf2-aligned-bh-spin-3.5pn/v1`.

| Component ID | Evidence | Reference expression | Frontends | Checks |
| --- | --- | --- | --- | --- |
| `tf2.bhspin.SO3.v1` | [wade2013](https://arxiv.org/pdf/1306.3901v2), Eq. 4 | [`spin_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); SO3 | [`torch.phase_coefficients[2]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[2]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S14](#spin-trace-s14), [S15](#spin-trace-s15), [S16](#spin-trace-s16), [S17](#spin-trace-s17) |
| `tf2.bhspin.SS4.v1` | [wade2013](https://arxiv.org/pdf/1306.3901v2), Eq. 4 | [`spin_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); SS4 | [`torch.phase_coefficients[3]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[3]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S14](#spin-trace-s14), [S15](#spin-trace-s15), [S16](#spin-trace-s16), [S17](#spin-trace-s17) |
| `tf2.bhspin.SO5.v1` | [wade2013](https://arxiv.org/pdf/1306.3901v2), Eq. 4 | [`spin_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); SO5 | [`torch.phase_coefficients[4]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[4]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S14](#spin-trace-s14), [S15](#spin-trace-s15), [S16](#spin-trace-s16), [S17](#spin-trace-s17) |
| `tf2.bhspin.SO5_log.v1` | [wade2013](https://arxiv.org/pdf/1306.3901v2), Eq. 4 | [`spin_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); SO5_log | [`torch.phase_coefficients[5]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[5]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S14](#spin-trace-s14), [S15](#spin-trace-s15), [S16](#spin-trace-s16), [S17](#spin-trace-s17) |
| `tf2.bhspin.SO6.v1` | [mishra2016](https://arxiv.org/pdf/1601.05588v1), Eq. 5, 6a, 7 | [`spin_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); SO6 | [`torch.phase_coefficients[6]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[6]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S14](#spin-trace-s14), [S15](#spin-trace-s15), [S16](#spin-trace-s16), [S17](#spin-trace-s17) |
| `tf2.bhspin.SS6.v1` | [mishra2016](https://arxiv.org/pdf/1601.05588v1), Eq. 5, 6a, 7 | [`spin_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); SS6 | [`torch.phase_coefficients[6]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[6]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S14](#spin-trace-s14), [S15](#spin-trace-s15), [S16](#spin-trace-s16), [S17](#spin-trace-s17) |
| `tf2.bhspin.SO7.v1` | [mishra2016](https://arxiv.org/pdf/1601.05588v1), Eq. 5, 6b, 7 | [`spin_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); SO7 | [`torch.phase_coefficients[8]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[8]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S14](#spin-trace-s14), [S15](#spin-trace-s15), [S16](#spin-trace-s16), [S17](#spin-trace-s17) |
| `tf2.bhspin.SS7.v1` | [pratten2020](https://arxiv.org/pdf/2001.11412v1), Eq. A19, A24 | [`spin_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); SS7 | [`torch.phase_coefficients[8]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[8]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S14](#spin-trace-s14), [S15](#spin-trace-s15), [S16](#spin-trace-s16), [S17](#spin-trace-s17) |
| `tf2.bhspin.SSS7.v1` | [mishra2016](https://arxiv.org/pdf/1601.05588v1), Eq. 5, 6b, 7 | [`spin_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); SSS7 | [`torch.phase_coefficients[8]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[8]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S14](#spin-trace-s14), [S15](#spin-trace-s15), [S16](#spin-trace-s16), [S17](#spin-trace-s17) |
| `tf2.bhspin.intrinsic_phase.v1` | Project convention `taylorf2-aligned-bh-spin-3.5pn/v1` | [`phase_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); composed intrinsic phase; explicit terms | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S18](#spin-trace-s18), [S19](#spin-trace-s19), [S20](#spin-trace-s20), [S21](#spin-trace-s21), [S22](#spin-trace-s22), [S23](#spin-trace-s23), [S24](#spin-trace-s24), [S17](#spin-trace-s17), [S25](#spin-trace-s25), [S26](#spin-trace-s26) |
| `tf2.bhspin.coalescence_phase.v1` | Project convention `taylorf2-aligned-bh-spin-3.5pn/v1` | [`waveform_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); phase; reference_frequency=False | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S18](#spin-trace-s18), [S19](#spin-trace-s19), [S20](#spin-trace-s20), [S21](#spin-trace-s21), [S22](#spin-trace-s22), [S23](#spin-trace-s23), [S24](#spin-trace-s24), [S17](#spin-trace-s17), [S25](#spin-trace-s25) |
| `tf2.bhspin.reference_phase.v1` | Project convention `taylorf2-aligned-bh-spin-3.5pn/v1` | [`waveform_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); phase; reference_frequency=True | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S18](#spin-trace-s18), [S19](#spin-trace-s19), [S20](#spin-trace-s20), [S21](#spin-trace-s21), [S22](#spin-trace-s22), [S23](#spin-trace-s23), [S24](#spin-trace-s24), [S17](#spin-trace-s17), [S25](#spin-trace-s25), [S27](#spin-trace-s27) |
| `tf2.bhspin.polarizations.v1` | Project convention `taylorf2-aligned-bh-spin-3.5pn/v1` | [`waveform_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); h_plus, h_cross; both phase modes | [`torch.evaluate`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S18](#spin-trace-s18), [S21](#spin-trace-s21), [S28](#spin-trace-s28), [S23](#spin-trace-s23), [S24](#spin-trace-s24), [S29](#spin-trace-s29), [S17](#spin-trace-s17), [S25](#spin-trace-s25), [S27](#spin-trace-s27), [S30](#spin-trace-s30) |

Unchanged inherited components: `taylorf2-nonspinning-3.5pn/v1`.

| Component ID | Evidence | Reference expression | Frontends | Checks |
| --- | --- | --- | --- | --- |
| `tf2.ns35.phi0.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi0 | [`torch.phase_coefficients[0]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[0]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S01](#spin-trace-s01) |
| `tf2.ns35.phi2.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi2 | [`torch.phase_coefficients[1]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[1]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S01](#spin-trace-s01) |
| `tf2.ns35.phi3.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi3 | [`torch.phase_coefficients[2]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[2]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S01](#spin-trace-s01) |
| `tf2.ns35.phi4.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi4 | [`torch.phase_coefficients[3]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[3]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S01](#spin-trace-s01) |
| `tf2.ns35.phi5.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi5 | [`torch.phase_coefficients[4]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[4]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S01](#spin-trace-s01) |
| `tf2.ns35.phi5_log.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi5_log | [`torch.phase_coefficients[5]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[5]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S01](#spin-trace-s01) |
| `tf2.ns35.phi6.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi6 | [`torch.phase_coefficients[6]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[6]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S01](#spin-trace-s01) |
| `tf2.ns35.phi6_log.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi6_log | [`torch.phase_coefficients[7]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[7]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S01](#spin-trace-s01) |
| `tf2.ns35.phi7.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi7 | [`torch.phase_coefficients[8]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[8]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S01](#spin-trace-s01) |
| `tf2.ns35.intrinsic_phase.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`phase_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); log(v) intrinsic phase | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S02](#spin-trace-s02), [S03](#spin-trace-s03), [S04](#spin-trace-s04), [S05](#spin-trace-s05), [S06](#spin-trace-s06), [S07](#spin-trace-s07) |
| `tf2.ns35.paper_phase.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`paper_phase_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); paper gauge; related adapters only | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) (related adapters; reference-only expression) | [S02](#spin-trace-s02) |
| `tf2.ns35.amplitude.v1` | [arun2008](https://arxiv.org/pdf/0810.5336v1), Eq. 6.13, D1 | [`amplitude_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); positive amplitude | [`torch.evaluate_amplitude`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_amplitude`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S08](#spin-trace-s08), [S03](#spin-trace-s03), [S04](#spin-trace-s04), [S05](#spin-trace-s05), [S06](#spin-trace-s06), [S07](#spin-trace-s07) |
| `tf2.ns35.polarizations.v1` | [arun2008](https://arxiv.org/pdf/0810.5336v1), Eq. 6.13, D1 | [`waveform_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); h_plus, h_cross; both phase modes | [`torch.evaluate`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S08](#spin-trace-s08), [S03](#spin-trace-s03), [S04](#spin-trace-s04), [S05](#spin-trace-s05), [S09](#spin-trace-s09), [S10](#spin-trace-s10), [S06](#spin-trace-s06), [S07](#spin-trace-s07), [S11](#spin-trace-s11), [S12](#spin-trace-s12) |
| `tf2.ns35.coalescence_phase.v1` | Project convention `taylorf2-nonspinning-3.5pn/v1` | [`waveform_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phase; reference_frequency=False | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S03](#spin-trace-s03), [S04](#spin-trace-s04), [S06](#spin-trace-s06), [S07](#spin-trace-s07), [S11](#spin-trace-s11), [S12](#spin-trace-s12) |
| `tf2.ns35.reference_phase.v1` | Project convention `taylorf2-nonspinning-3.5pn/v1` | [`waveform_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phase; reference_frequency=True | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [S02](#spin-trace-s02), [S03](#spin-trace-s03), [S04](#spin-trace-s04), [S13](#spin-trace-s13), [S06](#spin-trace-s06), [S07](#spin-trace-s07), [S11](#spin-trace-s11), [S12](#spin-trace-s12) |

Additional independent derivation:

| Component | Sources | Reference expression | Checks |
| --- | --- | --- | --- |
| `tf2.bhspin.SS7.v1` | [bohe2015](https://arxiv.org/pdf/1501.01529v1), Eq. 3.33, 4.14; [messina2017](https://arxiv.org/pdf/1703.08107v4), Eq. 23, 26; [buonanno2009-balance](https://arxiv.org/pdf/0907.0700v1), Eq. 3.2, 3.3a, 3.3b, 3.14 | [`ss7_balance_terms`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py) | [S31](#spin-trace-s31) |

Check labels are local to this table; function names identify maintained tests.

| Check | Maintained test |
| --- | --- |
| <span id="spin-trace-s01">S01</span> | [`test_pn_coefficients_against_exact_reference`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="spin-trace-s02">S02</span> | [`test_exact_equations_and_phase_gauge`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="spin-trace-s03">S03</span> | [`test_values_against_reference`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="spin-trace-s04">S04</span> | [`test_parameter_jacobian_against_symbolic_derivatives`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="spin-trace-s05">S05</span> | [`test_low_velocity_and_inclination_limits`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="spin-trace-s06">S06</span> | [`test_seeded_parameter_sweep`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="spin-trace-s07">S07</span> | [`test_single_precision_parameter_sweep`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="spin-trace-s08">S08</span> | [`test_newtonian_amplitude_reduction_and_polarization_limits`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="spin-trace-s09">S09</span> | [`test_nonspinning_symmetries`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="spin-trace-s10">S10</span> | [`test_compiled_waveform_against_reference`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="spin-trace-s11">S11</span> | [`test_batch_and_jax_jit_against_scalar_reference`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="spin-trace-s12">S12</span> | [`test_per_binary_frequency_grid_and_reference_vector`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="spin-trace-s13">S13</span> | [`test_reference_frequency_neighbors`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="spin-trace-s14">S14</span> | [`test_aligned_spin_coefficients_against_reference`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="spin-trace-s15">S15</span> | [`test_aligned_spin_corrections_by_spin_parity`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="spin-trace-s16">S16</span> | [`test_aligned_spin_selector_contract`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="spin-trace-s17">S17</span> | [`test_aligned_spin_zero_spin_reduction`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="spin-trace-s18">S18</span> | [`test_aligned_spin_values_against_reference`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="spin-trace-s19">S19</span> | [`test_aligned_spin_default_profiles`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="spin-trace-s20">S20</span> | [`test_aligned_spin_calibration_phase`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="spin-trace-s21">S21</span> | [`test_aligned_spin_parameter_jacobian`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="spin-trace-s22">S22</span> | [`test_aligned_spin_frequency_derivative`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="spin-trace-s23">S23</span> | [`test_aligned_spin_seeded_parameter_sweep`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="spin-trace-s24">S24</span> | [`test_aligned_spin_batch_and_compilation`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="spin-trace-s25">S25</span> | [`test_aligned_spin_zero_spin_backend_values`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="spin-trace-s26">S26</span> | [`test_aligned_spin_reference_domains`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="spin-trace-s27">S27</span> | [`test_aligned_spin_reference_precision_convergence`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="spin-trace-s28">S28</span> | [`test_aligned_spin_label_exchange`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="spin-trace-s29">S29</span> | [`test_aligned_spin_extremal_spin_bounds`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="spin-trace-s30">S30</span> | [`test_aligned_spin_reference_artifacts_are_current`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="spin-trace-s31">S31</span> | [`test_aligned_spin_spa_spin7_balance_factor`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |

<!-- END GENERATED ALIGNED SPIN TRACEABILITY -->

Metadata tests resolve the links and detect stale tables. They establish
structural coverage, not transcription correctness or numerical agreement.

## Numerical checks and limitations

The maintained FP64 checks compare both frontends with scalar reference
values at 80 decimal digits. They cover all coefficient tuple slots, selectors
`0, 2, 3, 4, 5, 6, 7, 8, -1`, the calibration exception, spin-degree isolation,
and public waveform profiles `5, 7, 8, -1` in both phase conventions. They also
check exact zero-spin reduction, body-label exchange, spin boundaries $\pm1$,
and first mass/spin Jacobians at zero and nonzero spins. Frequency derivatives
compare Torch's analytic derivative and JAX autodiff with symbolic differentiation.
Unit conversion uses the separate [SI/IAU constants audit](units.md).

A reproducible sweep (seed `20260923`, 12 systems) draws total mass logarithmically
from $[2.7,100]M_\odot$, mass ratio logarithmically from $[1,20]$, component spins
from $[-0.95,0.95]$, distance logarithmically from $[10,2000]$ Mpc, inclination
from $[0,\pi]$, $t_c$ from $[-0.1,0.1]$ seconds, and $\phi_c$ from $[-\pi,\pi]$.
Both $v$ and $v_{\rm ref}$ lie in $[0.12,0.30]$. Fixed cases add equal masses,
asymmetric masses, aligned/opposed spins, and reference-frequency cancellation.
These are sampled engineering checks, not a calibrated validity region.

Forward checks require relative amplitude error at most $5\times10^{-14}$,
unwrapped phase agreement with relative tolerance $5\times10^{-14}$ and
absolute tolerance $2\times10^{-10}$ radians, and polarization error
$|h-h_{\rm ref}|/A<3\times10^{-10}$. Mass/spin Jacobians normalize polarization
rows by the reference amplitude and use relative and absolute tolerances
$2\times10^{-9}$. Frequency derivative tolerances are relative $2\times10^{-13}$
and absolute $2\times10^{-12}$. Coefficient and spin-degree checks have their own
stricter intermediate tolerances in the maintained tests.

Shared one-dimensional frequency grids and explicit per-binary $(B,1)$ grids
with vector reference frequencies are checked in eager and compiled execution.
JAX JIT and Torch `torch.compile(backend="aot_eager", fullgraph=True)` are covered;
the latter checks Dynamo/AOT graph capture, not Inductor code generation. This
extension adds no FP32 spin qualification.

Committed reference fixtures are generated at 100 and 160 decimal digits,
checked for precision convergence, and store 65 digits with inputs, sources,
term selection, conventions, and derivatives. A separate near-reference test at $f=f_{\rm ref}+10^{-22}$ Hz
also compares 100 and 160 digits, requiring 65-digit agreement. This working
precision accounts for the digits lost in phase subtraction and a nearly
edge-on cross polarization. Increasing precision checks arithmetic stability;
it cannot detect the same incorrect expression evaluated twice.

This qualification concerns the declared reference equations and sampled cases.
It supplies neither a uniform floating-point error proof nor physical validation
against numerical relativity. No new LAL comparison, Lean proof, or Mathematica
cross-check is implied. Accelerator qualification is explicit and separate from
the maintained CPU job.

### Recorded CPU qualification

On 2026-09-22, the combined required gate passed all 408 tests with no skips,
expected failures, or deselections: 222 aligned-spin numerical/symbolic checks,
68 aligned-spin traceability checks, and 118 inherited reference/gate checks.
The environment was macOS arm64, Python 3.13.7, SymPy 1.14.0, mpmath 1.3.0,
Torch 2.14.0, and JAX 0.11.2, with JAX x64 enabled and both frontends on CPU.
Generated artifacts and both traceability maps were current; the four frozen
nonspinning kernel/table/fixture/test files retained their pre-extension hashes.
Related TaylorF2 and IMRPhenomD regressions passed 25 tests; one optional external
comparison was skipped because LALSuite/PyCBC was unavailable. The strict docs
build and Ruff checks passed. These results qualify the scope above, including
the corrected optional SS7 term, without adding an external-reference claim.

## Reproduce and review changes

Install the reference and both frontends, then run the complete qualification
gate so inherited and new coverage execute together:

```bash
python -m pip install -e '.[reference,torch,jax,dev]'
python -m diffgw_reference.artifacts --check
python -m diffgw_reference.aligned_spin_artifacts --check
python -m diffgw_reference.aligned_spin_traceability --check
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest tests/test_*math_reference*.py --require-math-reference --reference-device=cpu -ra
```

The required mode fails on missing dependencies, missing or deselected test
modules, skips, expected failures, and unexecuted tests. The workflow's presence
does not by itself make it a required merge check.

Change mathematical component versions when their numerical meaning changes;
give a composed specification a new version when its included terms or
conventions change. Keep explicit inherited versions and the baseline artifacts.
Changes to coverage or a qualified function name need no mathematical version
bump. Regenerate the component map with
`python -m diffgw_reference.aligned_spin_traceability` and review the resulting
diff alongside source evidence. Resolve disagreements against the publications
and documented conventions, without choosing formulas to fit backend outputs.
