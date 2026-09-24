# TaylorF2 adiabatic quadrupolar tides

## Scope and status

Specification: **`taylorf2-adiabatic-quadrupolar-tides-6.5pn/v1`**.
The independent [reference kernel](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_tidal.py)
adds the 5PN and 6PN electric quadrupole tidal phase and its 6.5PN tail to the
unchanged [nonspinning](taylorf2.md) and [aligned-spin](taylorf2-aligned-spin.md)
specifications. Each inherited component keeps its original ID and version.
The [generated coefficient table](taylorf2-tidal-coefficients.md) displays the
component coefficients and effective tidal parameters. Their independent
effective-parameter coefficient forms appear below.

This is a restricted-amplitude, circular, adiabatic inspiral specification.
It excludes magnetic and octupolar tides, dynamical tides and resonances,
nonlinear tides, internal tidal dissipation, spin-tidal couplings, tidal amplitude
corrections, merger, cutoffs, and tapers. The inherited spin sector retains
black-hole spin-induced multipoles: adding these tidal terms does **not**
produce a complete spinning neutron-star model. No equation-of-state relation
between masses, spins, and deformabilities is imposed.

## Inputs, conventions, and options

### Coordinates, units, and phase

Write \(M=m_1+m_2\), \(\delta=(m_1-m_2)/M\),
\(X_{1,2}=(1\pm\delta)/2\), and \(\eta=(1-\delta^2)/4\).
The inputs named `lambda1` and `lambda2` are the dimensionless individual
deformabilities \(\Lambda_i\), not the papers' dimensionful polarizabilities.
In geometric units, \(\lambda_i/M^5=X_i^5\Lambda_i\).

The numerical reference requires finite \(\Lambda_i\geq0\),
\(|\delta|<1\), \(|\chi_i|\leq1\), and positive frequency, mass in seconds,
distance in seconds, and reference frequency when supplied. Inclination and
phase angles are in radians; time shifts are in seconds. The frontend boundary
converts solar masses and Mpc using the independently [audited unit constants](units.md).
These domains define the expression and input checks; they do not establish
physical accuracy throughout the domain.

### Frontend option contract

The mathematical API selects `tidal_terms=("T10", "T12", "T13")` and
`spin_terms=` independently. Both default to their full reference term sets.
The frontend adapter
[`frontend_terms`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/tidal_traceability.py)
records integer `tidal_order` and the presence of each deformability:

<!-- BEGIN GENERATED TIDAL PROFILES -->

| `tidal_order` | Both deformabilities supplied | Either is `None` |
| --- | --- | --- |
| 0 through 9 | None | None |
| 10, 11 | `T10` | None |
| 12 (default) | `T10`, `T12` | None |
| 13 or greater | `T10`, `T12`, `T13` | None |
| negative | `T10`, `T12`, `T13` | None |

<!-- END GENERATED TIDAL PROFILES -->

Both frontends default to `tidal_order=12` in their coefficient, phase, and
public waveform APIs. Neither argument may be omitted to retain a one-body
tide: supply a numerical zero for the nondeformable body. `None` disables the
entire tidal contribution even when the other body has nonzero deformability.
The mathematical reference instead requires explicit numerical inputs.

Spin selection keeps the [existing documented profiles](taylorf2-aligned-spin.md#frontend-option-contract):
public waveforms default to `spin_order=7`, low-level phase to `spin_order=5`,
and all reference spin terms require `spin_order=8`. The reference's default
full composition therefore differs from the frontend defaults in two explicit
selectors. Tests use the corresponding adapter for each sector.

## Mathematics and components

### Tidal phase and coefficients

For \(v=(\pi M_{\rm sec} f)^{1/3}\), the added intrinsic phase is

\[
 I_{\rm tidal}(v)=\frac{3}{128\eta v^5}
   \left(T_{10}v^{10}+T_{12}v^{12}+T_{13}v^{13}\right).
\]

The canonical component form is

\[
\begin{aligned}
T_{10}&=-24\sum_i\Lambda_i X_i^4(12-11X_i),\\
T_{12}&=-\frac5{28}\sum_i\Lambda_i X_i^4
       (3179-919X_i-2286X_i^2+260X_i^3),\\
T_{13}&=-\pi T_{10}.
\end{aligned}
\]

The equivalent effective-parameter form is

\[
T_{10}=-\frac{39}{2}\widetilde\Lambda,\qquad
T_{12}=-\frac{3115}{64}\widetilde\Lambda
       +\frac{6595}{364}\delta\,\delta\widetilde\Lambda,\qquad
T_{13}=\frac{39\pi}{2}\widetilde\Lambda.
\]

The two effective parameters are defined in
[Wade et al., 1402.5156v1, Eqs. (5)–(6)](https://arxiv.org/pdf/1402.5156v1#page=4)
and displayed in the generated table. Their ordered-mass square root becomes
signed \(\delta\), extending the same component polynomials to either body
labelling. Under simultaneous label exchange, \(\widetilde\Lambda\) is even
and \(\delta\widetilde\Lambda\) is odd, so the phase is invariant and smooth
at equal mass. Equal masses with equal deformability give
\(\widetilde\Lambda=\Lambda\) and \(\delta\widetilde\Lambda=0\).

Both existing phase conventions and the \(e^{-i\Psi}\) carrier are retained.
The coalescence convention adds \(I_{\rm tidal}(f)\); the finite-reference
convention adds \(I_{\rm tidal}(f)-I_{\rm tidal}(f_{\rm ref})\).
The Newtonian amplitude and polarization prefactors are inherited unchanged.
Canonical derivatives hold every other input fixed, including both individual
deformabilities; they do not differentiate an unprovided equation of state.

### Direct paper audit

The audit read the pinned PDF equations directly, independently of frontend
outputs and generated fixtures. Wade's Eq. (A26), with
\(\lambda_i/M^5=X_i^5\Lambda_i\), gives the displayed component polynomials;
exact symbolic expansion agrees with Eqs. (4)–(6).
[Wade et al., 1402.5156v1, Eq. (A26)](https://arxiv.org/pdf/1402.5156v1#page=13)

As a separate normalization check, Damour–Nagar–Villain's per-body phase uses
\(\kappa_i=3(1-X_i)X_i^4\Lambda_i\). Substitution in Eqs. (B10)–(B11)
reproduces the same 5PN/6PN coefficients. Equation (B12) supplies the relative
factor \(-\pi v^3\), giving the positive effective \(T_{13}\) above for
arbitrary mass ratio. The retained terms do not depend on the unknown higher
order quantities elsewhere in that paper.
[Damour–Nagar–Villain, 1203.4352v1, Eqs. (B8)–(B14)](https://arxiv.org/pdf/1203.4352v1#page=26)

Seven independently transcribed exact identities checked the two papers'
5PN/6PN forms, tail normalization/sign, and signed-label transformations;
no coefficient disagreement was found. The maintained tests additionally
compare the kernel's component and effective forms exactly. The papers use
different printed Fourier exponential signs (Wade A23, Damour–Nagar–Villain 15);
this specification takes their phase polynomials and explicitly keeps the
previously audited project carrier and phase gauges.

## Implementation mapping

### Component traceability

The generated map records every new and inherited component, paper location,
reference function, both frontend functions, and maintained check names.
Tuple positions are explicit and checked against frontend source. The inherited
[spin-tail derivation](taylorf2-aligned-spin.md#component-traceability) also remains
in force. Structural tests resolve links without importing production frontends
and detect stale rendered option/component tables.

<!-- BEGIN GENERATED TIDAL TRACEABILITY -->

New components: `taylorf2-adiabatic-quadrupolar-tides-6.5pn/v1`.

| Component ID | Evidence | Reference expression | Frontends | Checks |
| --- | --- | --- | --- | --- |
| `tf2.tidal65.T10.v1` | [wade2014](https://arxiv.org/pdf/1402.5156v1), Eq. A26 | [`tidal_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_tidal.py); T10 | [`torch.phase_coefficients[9]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[9]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T31](#tidal-trace-t31), [T32](#tidal-trace-t32), [T33](#tidal-trace-t33), [T34](#tidal-trace-t34) |
| `tf2.tidal65.T12.v1` | [wade2014](https://arxiv.org/pdf/1402.5156v1), Eq. A26 | [`tidal_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_tidal.py); T12 | [`torch.phase_coefficients[10]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[10]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T31](#tidal-trace-t31), [T32](#tidal-trace-t32), [T33](#tidal-trace-t33), [T34](#tidal-trace-t34) |
| `tf2.tidal65.T13.v1` | [damour2012](https://arxiv.org/pdf/1203.4352v1), Eq. 6, B8, B9, B10, B12, B14 | [`tidal_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_tidal.py); T13 | [`torch.phase_coefficients[11]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[11]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T31](#tidal-trace-t31), [T32](#tidal-trace-t32), [T33](#tidal-trace-t33), [T34](#tidal-trace-t34) |
| `tf2.tidal65.lambda_tilde.v1` | [wade2014](https://arxiv.org/pdf/1402.5156v1), Eq. 5 | [`tidal_parameters`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_tidal.py); lambda_tilde; frontend tuple[0] | [`torch.tidal_combinations`](https://github.com/xangma/diffgw/blob/main/src/torchwave/conversions.py); [`jax.tidal_combinations`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/conversions.py) | [T35](#tidal-trace-t35), [T34](#tidal-trace-t34), [T36](#tidal-trace-t36), [T37](#tidal-trace-t37) |
| `tf2.tidal65.delta_lambda_tilde.v1` | [wade2014](https://arxiv.org/pdf/1402.5156v1), Eq. 6 | [`tidal_parameters`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_tidal.py); delta_lambda_tilde; frontend tuple[1] | [`torch.tidal_combinations`](https://github.com/xangma/diffgw/blob/main/src/torchwave/conversions.py); [`jax.tidal_combinations`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/conversions.py) | [T35](#tidal-trace-t35), [T34](#tidal-trace-t34), [T36](#tidal-trace-t36), [T37](#tidal-trace-t37) |
| `tf2.tidal65.intrinsic_phase.v1` | Project convention `taylorf2-adiabatic-quadrupolar-tides-6.5pn/v1` | [`phase_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_tidal.py); inherited phase plus selected tides | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T38](#tidal-trace-t38), [T39](#tidal-trace-t39), [T40](#tidal-trace-t40), [T41](#tidal-trace-t41), [T42](#tidal-trace-t42), [T37](#tidal-trace-t37), [T33](#tidal-trace-t33), [T34](#tidal-trace-t34) |
| `tf2.tidal65.coalescence_phase.v1` | Project convention `taylorf2-adiabatic-quadrupolar-tides-6.5pn/v1` | [`waveform_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_tidal.py); phase; reference_frequency=False | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T38](#tidal-trace-t38), [T39](#tidal-trace-t39), [T40](#tidal-trace-t40), [T41](#tidal-trace-t41), [T42](#tidal-trace-t42), [T37](#tidal-trace-t37), [T33](#tidal-trace-t33), [T34](#tidal-trace-t34) |
| `tf2.tidal65.reference_phase.v1` | Project convention `taylorf2-adiabatic-quadrupolar-tides-6.5pn/v1` | [`waveform_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_tidal.py); phase; reference_frequency=True | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T38](#tidal-trace-t38), [T39](#tidal-trace-t39), [T40](#tidal-trace-t40), [T41](#tidal-trace-t41), [T42](#tidal-trace-t42), [T37](#tidal-trace-t37), [T33](#tidal-trace-t33), [T34](#tidal-trace-t34), [T36](#tidal-trace-t36), [T43](#tidal-trace-t43), [T44](#tidal-trace-t44) |
| `tf2.tidal65.polarizations.v1` | Project convention `taylorf2-adiabatic-quadrupolar-tides-6.5pn/v1` | [`waveform_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_tidal.py); h_plus, h_cross; both phase conventions | [`torch.evaluate`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T38](#tidal-trace-t38), [T39](#tidal-trace-t39), [T40](#tidal-trace-t40), [T41](#tidal-trace-t41), [T42](#tidal-trace-t42), [T37](#tidal-trace-t37), [T33](#tidal-trace-t33), [T34](#tidal-trace-t34), [T36](#tidal-trace-t36), [T43](#tidal-trace-t43), [T44](#tidal-trace-t44) |

Unchanged inherited components: `taylorf2-nonspinning-3.5pn/v1`.

| Component ID | Evidence | Reference expression | Frontends | Checks |
| --- | --- | --- | --- | --- |
| `tf2.ns35.phi0.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi0 | [`torch.phase_coefficients[0]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[0]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T01](#tidal-trace-t01) |
| `tf2.ns35.phi2.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi2 | [`torch.phase_coefficients[1]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[1]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T01](#tidal-trace-t01) |
| `tf2.ns35.phi3.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi3 | [`torch.phase_coefficients[2]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[2]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T01](#tidal-trace-t01) |
| `tf2.ns35.phi4.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi4 | [`torch.phase_coefficients[3]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[3]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T01](#tidal-trace-t01) |
| `tf2.ns35.phi5.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi5 | [`torch.phase_coefficients[4]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[4]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T01](#tidal-trace-t01) |
| `tf2.ns35.phi5_log.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi5_log | [`torch.phase_coefficients[5]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[5]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T01](#tidal-trace-t01) |
| `tf2.ns35.phi6.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi6 | [`torch.phase_coefficients[6]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[6]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T01](#tidal-trace-t01) |
| `tf2.ns35.phi6_log.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi6_log | [`torch.phase_coefficients[7]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[7]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T01](#tidal-trace-t01) |
| `tf2.ns35.phi7.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi7 | [`torch.phase_coefficients[8]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[8]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T01](#tidal-trace-t01) |
| `tf2.ns35.intrinsic_phase.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`phase_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); log(v) intrinsic phase | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T02](#tidal-trace-t02), [T03](#tidal-trace-t03), [T04](#tidal-trace-t04), [T05](#tidal-trace-t05), [T06](#tidal-trace-t06), [T07](#tidal-trace-t07) |
| `tf2.ns35.paper_phase.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`paper_phase_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); paper gauge; related adapters only | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) (related adapters; reference-only expression) | [T02](#tidal-trace-t02) |
| `tf2.ns35.amplitude.v1` | [arun2008](https://arxiv.org/pdf/0810.5336v1), Eq. 6.13, D1 | [`amplitude_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); positive amplitude | [`torch.evaluate_amplitude`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_amplitude`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T08](#tidal-trace-t08), [T03](#tidal-trace-t03), [T04](#tidal-trace-t04), [T05](#tidal-trace-t05), [T06](#tidal-trace-t06), [T07](#tidal-trace-t07) |
| `tf2.ns35.polarizations.v1` | [arun2008](https://arxiv.org/pdf/0810.5336v1), Eq. 6.13, D1 | [`waveform_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); h_plus, h_cross; both phase modes | [`torch.evaluate`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T08](#tidal-trace-t08), [T03](#tidal-trace-t03), [T04](#tidal-trace-t04), [T05](#tidal-trace-t05), [T09](#tidal-trace-t09), [T10](#tidal-trace-t10), [T06](#tidal-trace-t06), [T07](#tidal-trace-t07), [T11](#tidal-trace-t11), [T12](#tidal-trace-t12) |
| `tf2.ns35.coalescence_phase.v1` | Project convention `taylorf2-nonspinning-3.5pn/v1` | [`waveform_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phase; reference_frequency=False | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T03](#tidal-trace-t03), [T04](#tidal-trace-t04), [T06](#tidal-trace-t06), [T07](#tidal-trace-t07), [T11](#tidal-trace-t11), [T12](#tidal-trace-t12) |
| `tf2.ns35.reference_phase.v1` | Project convention `taylorf2-nonspinning-3.5pn/v1` | [`waveform_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phase; reference_frequency=True | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T02](#tidal-trace-t02), [T03](#tidal-trace-t03), [T04](#tidal-trace-t04), [T13](#tidal-trace-t13), [T06](#tidal-trace-t06), [T07](#tidal-trace-t07), [T11](#tidal-trace-t11), [T12](#tidal-trace-t12) |

Unchanged inherited components: `taylorf2-aligned-bh-spin-3.5pn/v1`.

| Component ID | Evidence | Reference expression | Frontends | Checks |
| --- | --- | --- | --- | --- |
| `tf2.bhspin.SO3.v1` | [wade2013](https://arxiv.org/pdf/1306.3901v2), Eq. 4 | [`spin_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); SO3 | [`torch.phase_coefficients[2]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[2]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T14](#tidal-trace-t14), [T15](#tidal-trace-t15), [T16](#tidal-trace-t16), [T17](#tidal-trace-t17) |
| `tf2.bhspin.SS4.v1` | [wade2013](https://arxiv.org/pdf/1306.3901v2), Eq. 4 | [`spin_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); SS4 | [`torch.phase_coefficients[3]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[3]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T14](#tidal-trace-t14), [T15](#tidal-trace-t15), [T16](#tidal-trace-t16), [T17](#tidal-trace-t17) |
| `tf2.bhspin.SO5.v1` | [wade2013](https://arxiv.org/pdf/1306.3901v2), Eq. 4 | [`spin_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); SO5 | [`torch.phase_coefficients[4]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[4]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T14](#tidal-trace-t14), [T15](#tidal-trace-t15), [T16](#tidal-trace-t16), [T17](#tidal-trace-t17) |
| `tf2.bhspin.SO5_log.v1` | [wade2013](https://arxiv.org/pdf/1306.3901v2), Eq. 4 | [`spin_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); SO5_log | [`torch.phase_coefficients[5]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[5]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T14](#tidal-trace-t14), [T15](#tidal-trace-t15), [T16](#tidal-trace-t16), [T17](#tidal-trace-t17) |
| `tf2.bhspin.SO6.v1` | [mishra2016](https://arxiv.org/pdf/1601.05588v1), Eq. 5, 6a, 7 | [`spin_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); SO6 | [`torch.phase_coefficients[6]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[6]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T14](#tidal-trace-t14), [T15](#tidal-trace-t15), [T16](#tidal-trace-t16), [T17](#tidal-trace-t17) |
| `tf2.bhspin.SS6.v1` | [mishra2016](https://arxiv.org/pdf/1601.05588v1), Eq. 5, 6a, 7 | [`spin_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); SS6 | [`torch.phase_coefficients[6]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[6]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T14](#tidal-trace-t14), [T15](#tidal-trace-t15), [T16](#tidal-trace-t16), [T17](#tidal-trace-t17) |
| `tf2.bhspin.SO7.v1` | [mishra2016](https://arxiv.org/pdf/1601.05588v1), Eq. 5, 6b, 7 | [`spin_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); SO7 | [`torch.phase_coefficients[8]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[8]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T14](#tidal-trace-t14), [T15](#tidal-trace-t15), [T16](#tidal-trace-t16), [T17](#tidal-trace-t17) |
| `tf2.bhspin.SS7.v1` | [pratten2020](https://arxiv.org/pdf/2001.11412v1), Eq. A19, A24 | [`spin_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); SS7 | [`torch.phase_coefficients[8]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[8]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T14](#tidal-trace-t14), [T15](#tidal-trace-t15), [T16](#tidal-trace-t16), [T17](#tidal-trace-t17) |
| `tf2.bhspin.SSS7.v1` | [mishra2016](https://arxiv.org/pdf/1601.05588v1), Eq. 5, 6b, 7 | [`spin_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); SSS7 | [`torch.phase_coefficients[8]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[8]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T14](#tidal-trace-t14), [T15](#tidal-trace-t15), [T16](#tidal-trace-t16), [T17](#tidal-trace-t17) |
| `tf2.bhspin.intrinsic_phase.v1` | Project convention `taylorf2-aligned-bh-spin-3.5pn/v1` | [`phase_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); composed intrinsic phase; explicit terms | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T18](#tidal-trace-t18), [T19](#tidal-trace-t19), [T20](#tidal-trace-t20), [T21](#tidal-trace-t21), [T22](#tidal-trace-t22), [T23](#tidal-trace-t23), [T24](#tidal-trace-t24), [T17](#tidal-trace-t17), [T25](#tidal-trace-t25), [T26](#tidal-trace-t26) |
| `tf2.bhspin.coalescence_phase.v1` | Project convention `taylorf2-aligned-bh-spin-3.5pn/v1` | [`waveform_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); phase; reference_frequency=False | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T18](#tidal-trace-t18), [T19](#tidal-trace-t19), [T20](#tidal-trace-t20), [T21](#tidal-trace-t21), [T22](#tidal-trace-t22), [T23](#tidal-trace-t23), [T24](#tidal-trace-t24), [T17](#tidal-trace-t17), [T25](#tidal-trace-t25) |
| `tf2.bhspin.reference_phase.v1` | Project convention `taylorf2-aligned-bh-spin-3.5pn/v1` | [`waveform_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); phase; reference_frequency=True | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T18](#tidal-trace-t18), [T19](#tidal-trace-t19), [T20](#tidal-trace-t20), [T21](#tidal-trace-t21), [T22](#tidal-trace-t22), [T23](#tidal-trace-t23), [T24](#tidal-trace-t24), [T17](#tidal-trace-t17), [T25](#tidal-trace-t25), [T27](#tidal-trace-t27) |
| `tf2.bhspin.polarizations.v1` | Project convention `taylorf2-aligned-bh-spin-3.5pn/v1` | [`waveform_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_aligned_spin.py); h_plus, h_cross; both phase modes | [`torch.evaluate`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T18](#tidal-trace-t18), [T21](#tidal-trace-t21), [T28](#tidal-trace-t28), [T23](#tidal-trace-t23), [T24](#tidal-trace-t24), [T29](#tidal-trace-t29), [T17](#tidal-trace-t17), [T25](#tidal-trace-t25), [T27](#tidal-trace-t27), [T30](#tidal-trace-t30) |

Additional reference expressions:

| Expression | Evidence | Reference function | Checks |
| --- | --- | --- | --- |
| Effective-parameter coefficients: T10, T12, T13 | [wade2014](https://arxiv.org/pdf/1402.5156v1), Eq. 4, 5, 6, A26; [damour2012](https://arxiv.org/pdf/1203.4352v1), Eq. B12 | [`effective_tidal_corrections`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_tidal.py) | [T34](#tidal-trace-t34), [T31](#tidal-trace-t31) |
| Isolated additive tidal phase: 3/(128 eta) times selected v powers | [wade2014](https://arxiv.org/pdf/1402.5156v1), Eq. A26; [damour2012](https://arxiv.org/pdf/1203.4352v1), Eq. 6, B8, B9, B10, B11, B12, B14 | [`tidal_phase_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_tidal.py) | [T34](#tidal-trace-t34), [T38](#tidal-trace-t38), [T40](#tidal-trace-t40) |

Check labels are local to this table; function names identify maintained tests.

| Check | Maintained test |
| --- | --- |
| <span id="tidal-trace-t01">T01</span> | [`test_pn_coefficients_against_exact_reference`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="tidal-trace-t02">T02</span> | [`test_exact_equations_and_phase_gauge`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="tidal-trace-t03">T03</span> | [`test_values_against_reference`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="tidal-trace-t04">T04</span> | [`test_parameter_jacobian_against_symbolic_derivatives`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="tidal-trace-t05">T05</span> | [`test_low_velocity_and_inclination_limits`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="tidal-trace-t06">T06</span> | [`test_seeded_parameter_sweep`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="tidal-trace-t07">T07</span> | [`test_single_precision_parameter_sweep`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="tidal-trace-t08">T08</span> | [`test_newtonian_amplitude_reduction_and_polarization_limits`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="tidal-trace-t09">T09</span> | [`test_nonspinning_symmetries`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="tidal-trace-t10">T10</span> | [`test_compiled_waveform_against_reference`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="tidal-trace-t11">T11</span> | [`test_batch_and_jax_jit_against_scalar_reference`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="tidal-trace-t12">T12</span> | [`test_per_binary_frequency_grid_and_reference_vector`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="tidal-trace-t13">T13</span> | [`test_reference_frequency_neighbors`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="tidal-trace-t14">T14</span> | [`test_aligned_spin_coefficients_against_reference`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="tidal-trace-t15">T15</span> | [`test_aligned_spin_corrections_by_spin_parity`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="tidal-trace-t16">T16</span> | [`test_aligned_spin_selector_contract`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="tidal-trace-t17">T17</span> | [`test_aligned_spin_zero_spin_reduction`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="tidal-trace-t18">T18</span> | [`test_aligned_spin_values_against_reference`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="tidal-trace-t19">T19</span> | [`test_aligned_spin_default_profiles`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="tidal-trace-t20">T20</span> | [`test_aligned_spin_calibration_phase`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="tidal-trace-t21">T21</span> | [`test_aligned_spin_parameter_jacobian`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="tidal-trace-t22">T22</span> | [`test_aligned_spin_frequency_derivative`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="tidal-trace-t23">T23</span> | [`test_aligned_spin_seeded_parameter_sweep`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="tidal-trace-t24">T24</span> | [`test_aligned_spin_batch_and_compilation`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="tidal-trace-t25">T25</span> | [`test_aligned_spin_zero_spin_backend_values`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="tidal-trace-t26">T26</span> | [`test_aligned_spin_reference_domains`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="tidal-trace-t27">T27</span> | [`test_aligned_spin_reference_precision_convergence`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="tidal-trace-t28">T28</span> | [`test_aligned_spin_label_exchange`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="tidal-trace-t29">T29</span> | [`test_aligned_spin_extremal_spin_bounds`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="tidal-trace-t30">T30</span> | [`test_aligned_spin_reference_artifacts_are_current`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_aligned_spin_math_reference.py) |
| <span id="tidal-trace-t31">T31</span> | [`test_tidal_coefficients_against_reference`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_tidal_math_reference.py) |
| <span id="tidal-trace-t32">T32</span> | [`test_tidal_selector_contract`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_tidal_math_reference.py) |
| <span id="tidal-trace-t33">T33</span> | [`test_tidal_default_and_missing_arguments`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_tidal_math_reference.py) |
| <span id="tidal-trace-t34">T34</span> | [`test_tidal_zero_reduction_and_exchange`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_tidal_math_reference.py) |
| <span id="tidal-trace-t35">T35</span> | [`test_tidal_combinations_against_reference`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_tidal_math_reference.py) |
| <span id="tidal-trace-t36">T36</span> | [`test_tidal_label_exchange`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_tidal_math_reference.py) |
| <span id="tidal-trace-t37">T37</span> | [`test_tidal_transformed_jacobian`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_tidal_math_reference.py) |
| <span id="tidal-trace-t38">T38</span> | [`test_tidal_values_against_reference`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_tidal_math_reference.py) |
| <span id="tidal-trace-t39">T39</span> | [`test_tidal_parameter_jacobian`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_tidal_math_reference.py) |
| <span id="tidal-trace-t40">T40</span> | [`test_tidal_frequency_derivative`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_tidal_math_reference.py) |
| <span id="tidal-trace-t41">T41</span> | [`test_tidal_seeded_parameter_sweep`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_tidal_math_reference.py) |
| <span id="tidal-trace-t42">T42</span> | [`test_tidal_batch_and_compilation`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_tidal_math_reference.py) |
| <span id="tidal-trace-t43">T43</span> | [`test_tidal_reference_domains_and_convergence`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_tidal_math_reference.py) |
| <span id="tidal-trace-t44">T44</span> | [`test_tidal_reference_artifacts_are_current`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_tidal_math_reference.py) |

<!-- END GENERATED TIDAL TRACEABILITY -->

## Evidence and limitations

The [maintained LAL binary check](../validation.md#taylorf2-electric-tides-against-lalsuite-23-september-2026)
also compares both frontends with LAL at tidal orders 10, 12, and 13 for
nonspinning equal- and unequal-mass binaries. It does not qualify spinning
tidal LAL parity: the tested LAL call has an additional spin-dependent response
to nonzero deformabilities even when its tidal PN selector is zero.

### Numerical qualification

On 22 September 2026, the complete required-reference gate passed **576/576
tests**, with no skips, expected failures, or deselection. This includes 82
tidal numerical/symbolic tests and 86 tidal traceability tests, alongside the
408 existing reference checks. The run used CPU execution on macOS 26.6 ARM64,
Python 3.13.7, SymPy 1.14.0, mpmath 1.3.0, Torch 2.14.0, and JAX 0.11.2
with JAX x64 enabled. The tidal comparisons qualify FP64; GPU and FP32 tidal
execution remain unqualified. All 17 pre-existing reference source, test,
fixture, and coefficient-table files remained byte-for-byte unchanged.

Maintained FP64 checks compare both frontends against high-precision scalar
evaluation for effective parameters, individual tidal coefficients, unwrapped
phase, polarizations, and derivatives with respect to mass, deformability,
and frequency. They also exercise selector boundaries, missing inputs, zero and
one-body tides, body exchange, equal mass, seeded systems, shared and per-binary
frequency grids, and both phase conventions. Symbolic derivatives in canonical
coordinates are transformed explicitly to frontend coordinates.

The seed `20260924` supplies ten systems alternating component masses in
\([1,2.2]M_\odot\) with one-body-tide systems whose first mass is
\([3,12]M_\odot\) and second is \([1,2.2]M_\odot\). Sampled deformabilities
lie in \([0,2000]\), with `lambda1=0` in the latter systems;
\(\chi_1\in[-0.5,0.5]\), \(\chi_2\in[-0.1,0.1]\), and both \(v\) and
\(v_{\rm ref}\) lie in \([0.12,0.30]\). These are engineering checks of
the specified composition, not an astrophysical population or neutron-star
equation of state. Fixed cases include \(f=f_{\rm ref}\) and adjacent FP64
frequencies. A separate 100/160-digit convergence comparison uses
\(f=f_{\rm ref}+10^{-22}\).

| Compared quantity | FP64 budget |
| --- | --- |
| Tidal coefficients | Relative \(2\times10^{-14}\), absolute \(2\times10^{-13}\) |
| Effective parameters | Relative \(2\times10^{-14}\), absolute \(2\times10^{-12}\) |
| Amplitude | Relative \(5\times10^{-14}\) |
| Unwrapped phase | Relative \(5\times10^{-14}\), absolute \(2\times10^{-10}\) rad |
| Each complex polarization | \(\lvert h-h_{\rm ref}\rvert/A<3\times10^{-10}\) |
| Frequency derivative of phase | Relative \(2\times10^{-13}\), absolute \(2\times10^{-12}\) |
| Scaled parameter Jacobian | Relative and absolute \(2\times10^{-9}\) |

For the parameter Jacobian, amplitude and polarization rows are divided by the
reference amplitude; the phase row is unscaled. Columns for
\((m_1,m_2,\Lambda_1,\Lambda_2)\) are multiplied by \((1,1,1000,1000)\),
so the absolute budget per unit deformability is \(2\times10^{-12}\) in
these normalized coordinates. This fixed scale remains defined at zero tide.
All budgets are engineering tolerances, not interval bounds or physical error
estimates.

Compilation checks use JAX JIT and Torch full-graph `aot_eager`; the latter
qualifies graph capture and AOT execution, not Inductor code generation.
Cases include a shared one-dimensional grid, explicit per-binary `(B,1)` and
`(B,Nf)` grids, vector reference frequencies, and compiled vectorized Jacobians.
The exact cases and assertions remain in the maintained test module.
The tests do not qualify spinning/tidal FP32 evaluation, GPU
execution unless explicitly run there, or physical waveform error against
numerical relativity.

Frozen high-precision values and their precision-convergence check are generated
from this kernel. They detect unintended changes but are not independent
scientific evidence. Exact identities, source review, and frontend comparisons
provide different checks; none makes this reference infallible.

## Reproduce the checks

From the repository root after installing `.[reference,torch,jax,dev]`:

```sh
python -m diffgw_reference.tidal_artifacts --check
python -m diffgw_reference.tidal_traceability --check
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest tests/test_*math_reference*.py --require-math-reference --reference-device=cpu -ra
```

The complete required-mode run fails on missing dependencies or reference
modules, skipped or expected-failure cases, and deselection. A focused local
run of `tests/test_taylorf2_tidal_math_reference.py` and
`tests/test_tidal_math_reference_traceability.py` is useful during development;
it does not replace the complete CPU qualification.

A source, convention, or equation change that changes mathematical meaning
requires a new specification/component version, an explained difference,
reviewed artifact regeneration, and updated traceability and coverage. This
extension leaves prior nonspinning and aligned-spin equations, APIs, and
artifacts unchanged.
