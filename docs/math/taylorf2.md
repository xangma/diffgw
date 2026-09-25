# TaylorF2 mathematical specification

<!-- BEGIN GENERATED WAVEFORM DEFINITION -->

| Definition field | Catalog value |
| --- | --- |
| Model ID | `TaylorF2` |
| Reference composition | `taylorf2-nonspinning-3.5pn/v1` |
| Scope | Nonspinning 3.5PN inspiral with restricted quadrupole amplitude and positive frequencies. |
| Domain and modes | `frequency`; `(2, 2)` |
| Native backends | `torch`, `jax` |
| Scoped verification claims | 0 |

| Option | Default | Native choices | Reference-only choices |
| --- | --- | --- | --- |
| `phasing_convention` | `coalescence` | `coalescence`, `reference_frequency` | None |

<!-- END GENERATED WAVEFORM DEFINITION -->

## Scope and status

This first specification covers quasi-circular, nonspinning point particles:
3.5PN intrinsic phase and restricted (Newtonian) amplitude. It defines positive
frequency polarizations and two explicit DiffGW phase conventions. It has no
calibration tables or imported companion code.

Generate `TaylorF2` with either backend through the
[frequency-domain API](../guides/waveform-generation.md). For the additional
physics available in the native model, read the separately versioned
[aligned-spin](taylorf2-aligned-spin.md), [tidal](taylorf2-tidal.md), and
[spin-induced quadrupole](taylorf2-spin-quadrupole.md) specifications. They
extend this base; `TaylorF2_HM` and `TaylorF2_Eccentric` have different scope.

Canonical implementation:
[`src/diffgw_reference/taylorf2.py`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py).
Symbols, `COMPONENT_IDS`, and `COMPONENT_SOURCES` are machine-readable in that module.
The maintained [component map](#component-traceability) links each of those IDs
to its expression, both frontend functions, and numerical or symbolic checks.

### Source evidence and scope

| Component | Primary source | Treatment |
| --- | --- | --- |
| `tf2.ns35.phi*.v1`, `tf2.ns35.intrinsic_phase.v1`, `tf2.ns35.paper_phase.v1` | [Buonanno et al., arXiv:0907.0700v1](https://arxiv.org/pdf/0907.0700v1), Eqs. (3.14), (3.17), (3.18) | Exact nonspinning coefficients; source logarithm scale retained separately; paper's $\nu$ becomes $\eta$ |
| `tf2.ns35.amplitude.v1`, `tf2.ns35.polarizations.v1` | [Arun et al., arXiv:0810.5336v1](https://arxiv.org/pdf/0810.5336v1), Eqs. (6.13), (D1) | Only $n=0,k=2$; remove the cutoff for the unwindowed expression; map Fourier and phase conventions explicitly |
| `tf2.ns35.coalescence_phase.v1`, `tf2.ns35.reference_phase.v1` | DiffGW specification choices, below | Phase origin, orbital phase parameter, and optional intrinsic-phase subtraction; not attributed verbatim to a paper equation |

Equation numbers refer to the pinned PDFs; arXiv's generated HTML can renumber
them. Later paper revisions or errata must be reviewed before extending this
scope. No spin or tidal terms are certified by this nonspinning transcription.

#### Direct source review

On 2026-09-22, a separate review compared the printed
[Buonanno Eq. (3.18), p. 8](https://arxiv.org/pdf/0907.0700v1#page=8)
directly with the SymPy kernel. An independently entered expression matched
`paper_phase_expr()` exactly, including all rational coefficients, the 2.5PN
logarithm scale, and the 3PN Euler constant and $\ln(4v)$ terms. The difference
from the chosen $\ln v$ gauge was independent of frequency, as stated below.

The review also reduced the $n=0,k=2$ term of
[Arun Eqs. (6.13), (D1), pp. 19 and 40](https://arxiv.org/pdf/0810.5336v1#page=19)
to the stated amplitude and checked both polarization signs after Fourier
conjugation. The extra coalescence-phase offset remains an explicit project
convention. No transcription correction was found in this bounded review.
This was a direct paper-to-expression check, independent of generated fixtures;
it was not a new derivation of the PN results or a formal proof.

The [aligned-spin extension](taylorf2-aligned-spin.md) composes this unchanged
version with separately sourced spin terms.

## Inputs, conventions, and options

| Symbol / argument | Meaning and units | Domain |
| --- | --- | --- |
| $M$ / `mass_seconds` | $G(m_1+m_2)/c^3$, seconds | $M>0$ |
| $\eta$ / `eta` | $m_1m_2/(m_1+m_2)^2$ | $0<\eta\leq1/4$ |
| $D$ / `distance_seconds` | Luminosity distance divided by $c$, seconds | $D>0$ |
| $f$ / `frequency` | GW frequency, Hz | $f>0$ |
| $\iota$ / `inclination` | Inclination, radians | $0\leq\iota\leq\pi$ |
| $t_c$ / `tc` | Time shift, seconds | Finite real |
| $\phi_c$ / `phic` | Orbital phase parameter, radians | Finite real |
| $f_{\rm ref}$ / `reference_frequency` | Optional phase reference, Hz | Absent or strictly positive |

All inputs must be finite. The reference rejects zero/negative frequencies and
does not apply the frontends' low-frequency clamp. Its analytic domain is broader
than the physical slow-inspiral regime of the PN approximation. Evaluation above
an inspiral termination frequency tests algebra only.

The catalog's `coalescence` default applies when `f_ref` is omitted. Both native
dispatchers select `reference_frequency` when a positive `f_ref` is supplied
without an explicit `phasing_convention`. For the cataloged nonspinning,
unwindowed waveform, call `diffgw.get_fd_waveform("TaylorF2", ...)` with zero
aligned spins (the native default), no tidal deformabilities, and
`f_isco_cutoff=False`. Supply a positive `f_ref` for reference-frequency mode;
otherwise omit it. The native waveform defaults to `f_isco_cutoff=True`, so an
ordinary call includes an ISCO window outside this reference composition.
The native fallback that anchors explicit reference-frequency mode to the first
sample when `f_ref` is absent is also outside this specification.

## Mathematics and components

### Intrinsic phase and exact coefficients

Let $v=(\pi M f)^{1/3}$. Component `tf2.ns35.intrinsic_phase.v1` is

$$
I(v,\eta)=\frac{3}{128\eta v^5}\left[
\sum_{k\in\{0,2,3,4,5,6,7\}}\varphi_k v^k
+\varphi_{5,\ln}v^5\ln v+\varphi_{6,\ln}v^6\ln(4v)
\right].
$$

The table is generated from `coefficients()`; `phi5_log` and `phi6_log` label
$\varphi_{5,\ln}$ and $\varphi_{6,\ln}$. $\gamma$ is the exact Euler constant.
Coefficients exclude the overall $3/(128\eta v^5)$ factor.

--8<-- "docs/math/taylorf2-coefficients.md"

The paper writes the 2.5PN logarithm as $\ln(v/v_{\rm lso})$, with
$v_{\rm lso}=1/\sqrt6$. Therefore `paper_phase_expr()` preserves

$$
I_{\rm paper}=I-\Delta(\eta),\qquad
\Delta(\eta)=\frac{9\varphi_5\ln v_{\rm lso}}{128\eta}.
$$

For equal total phase, the paper's GW phase constant maps to
$2\phi_{c,\log v}=\phi_{c,\rm paper}+\Delta(\eta)$.
This offset depends on mass ratio: derivatives at fixed paper phase and fixed
DiffGW `phic` are different. Finite-reference subtraction cancels it exactly.

### Amplitude, polarization, and phase conventions

The Newtonian $k=2$ factor in Arun Eqs. (6.13), (D1) reduces to
$M^2D^{-1}\sqrt{5\pi\eta/96}\,v^{-7/2}$ multiplying the angular factors.
Define the positive face-on amplitude

$$
A(f)=\sqrt{\frac{5\eta}{24}}\frac{M^{5/6}}{\pi^{2/3}D}f^{-7/6}.
$$

Its units are seconds (Fourier strain). The adapter uses
$\tilde h(f)=\int h(t)e^{-2\pi i ft}\,dt$ and

$$
\tilde h_+=\frac{1+\cos^2\iota}{2}A e^{-i\Psi},\qquad
\tilde h_\times=-i\cos\iota\,A e^{-i\Psi}.
$$

The source's positive-Fourier exponential is conjugated. Its negative plus
amplitude can be absorbed as a $\pi$ phase shift, yielding a $+3\pi/4$ constant.
The coalescence adapter chooses a further constant phase-origin offset, giving
$-\pi/4$. These are stated conventions, not additional PN physics.

| Mode | Definition of $\Psi$ |
| --- | --- |
| Coalescence (`reference_frequency=None`) | $I(f)+2\pi f t_c-2\phi_c-\pi/4$ |
| Reference frequency (explicit positive value) | $I(f)-I(f_{\rm ref})+2\pi f t_c-2\phi_c+3\pi/4$ |

Here $I(f)$ abbreviates $I((\pi Mf)^{1/3},\eta)$. In reference mode,
$\Psi(f_{\rm ref})=2\pi f_{\rm ref}t_c-2\phi_c+3\pi/4$; `phic` is the adapter's
phase parameter, not asserted to equal the instantaneous orbital phase there.
Comparisons use these conventions directly, without fitting an extra phase or
time shift to improve agreement.

## Implementation mapping

### Derivatives and frontend mapping

`derivative_expr(expression, symbol, order=1)` differentiates the canonical
expression at fixed values of the other inputs. For example,
$\partial_f I=(v/3f)\partial_v I$, $\partial_{t_c}\Psi=2\pi f$,
$\partial_{\phi_c}\Psi=-2$, and $\partial_D A=-A/D$.
The derivative with respect to $f_{\rm ref}$ includes the subtracted intrinsic
phase. Zero spin values do not validate spin derivatives.

Tests convert production solar masses and Mpc to $M,D$ using explicitly supplied
`diffgw_core` constants, then apply the analytic chain rule for mass and distance
derivatives. The math kernel has no dependency on those constants. The separate
[SI/IAU unit audit](units.md) checks `C_SI`, `MSUN_SEC`, and `MPC_SI` against
independent definitions through nearest FP64 rounding. It does not audit `G_SI`
and `MSUN_SI` individually or other physical constants. Cutoffs are disabled in
both frontends.

### Component traceability

The following map is generated from
[`diffgw_reference.traceability`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/traceability.py).
Its frontend names are metadata strings; importing the reference does not load
Torch, JAX, or production waveform helpers. Bracketed numbers are zero-based
positions in the explicitly named `phase_coefficients` return tuple, checked
against both frontends. The three following tidal positions must be zero in
these nonspinning point-particle tests; their tidal formulas are outside scope.

`paper_phase` is reference-only: both frontend links identify the adapters
whose gauge transformation is documented above. Neither frontend exposes that
paper-gauge expression directly. The symbolic gauge test checks the relation;
it does not claim direct paper-gauge frontend coverage. Intrinsic-phase
comparisons run through the explicitly specified frontend phase adapters.

<!-- BEGIN GENERATED TAYLORF2 TRACEABILITY -->

| Component ID | Evidence | Reference expression | Frontends | Checks |
| --- | --- | --- | --- | --- |
| `tf2.ns35.phi0.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi0 | [`torch.phase_coefficients[0]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[0]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T01](#trace-t01) |
| `tf2.ns35.phi2.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi2 | [`torch.phase_coefficients[1]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[1]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T01](#trace-t01) |
| `tf2.ns35.phi3.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi3 | [`torch.phase_coefficients[2]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[2]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T01](#trace-t01) |
| `tf2.ns35.phi4.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi4 | [`torch.phase_coefficients[3]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[3]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T01](#trace-t01) |
| `tf2.ns35.phi5.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi5 | [`torch.phase_coefficients[4]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[4]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T01](#trace-t01) |
| `tf2.ns35.phi5_log.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi5_log | [`torch.phase_coefficients[5]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[5]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T01](#trace-t01) |
| `tf2.ns35.phi6.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi6 | [`torch.phase_coefficients[6]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[6]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T01](#trace-t01) |
| `tf2.ns35.phi6_log.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi6_log | [`torch.phase_coefficients[7]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[7]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T01](#trace-t01) |
| `tf2.ns35.phi7.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`coefficients`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phi7 | [`torch.phase_coefficients[8]`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.phase_coefficients[8]`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T01](#trace-t01) |
| `tf2.ns35.intrinsic_phase.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`phase_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); log(v) intrinsic phase | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T02](#trace-t02), [T03](#trace-t03), [T04](#trace-t04), [T05](#trace-t05), [T06](#trace-t06), [T07](#trace-t07) |
| `tf2.ns35.paper_phase.v1` | [buonanno2009](https://arxiv.org/pdf/0907.0700v1), Eq. 3.18 | [`paper_phase_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); paper gauge; related adapters only | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) (related adapters; reference-only expression) | [T02](#trace-t02) |
| `tf2.ns35.amplitude.v1` | [arun2008](https://arxiv.org/pdf/0810.5336v1), Eq. 6.13, D1 | [`amplitude_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); positive amplitude | [`torch.evaluate_amplitude`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_amplitude`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T08](#trace-t08), [T03](#trace-t03), [T04](#trace-t04), [T05](#trace-t05), [T06](#trace-t06), [T07](#trace-t07) |
| `tf2.ns35.polarizations.v1` | [arun2008](https://arxiv.org/pdf/0810.5336v1), Eq. 6.13, D1 | [`waveform_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); h_plus, h_cross; both phase modes | [`torch.evaluate`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T08](#trace-t08), [T03](#trace-t03), [T04](#trace-t04), [T05](#trace-t05), [T09](#trace-t09), [T10](#trace-t10), [T06](#trace-t06), [T07](#trace-t07), [T11](#trace-t11), [T12](#trace-t12) |
| `tf2.ns35.coalescence_phase.v1` | Project convention `taylorf2-nonspinning-3.5pn/v1` | [`waveform_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phase; reference_frequency=False | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T03](#trace-t03), [T04](#trace-t04), [T06](#trace-t06), [T07](#trace-t07), [T11](#trace-t11), [T12](#trace-t12) |
| `tf2.ns35.reference_phase.v1` | Project convention `taylorf2-nonspinning-3.5pn/v1` | [`waveform_expr`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2.py); phase; reference_frequency=True | [`torch.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2.py); [`jax.evaluate_phase`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2.py) | [T02](#trace-t02), [T03](#trace-t03), [T04](#trace-t04), [T13](#trace-t13), [T06](#trace-t06), [T07](#trace-t07), [T11](#trace-t11), [T12](#trace-t12) |

Check labels below are local to this table; the function names identify the maintained tests.

| Check | Maintained test |
| --- | --- |
| <span id="trace-t01">T01</span> | [`test_pn_coefficients_against_exact_reference`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="trace-t02">T02</span> | [`test_exact_equations_and_phase_gauge`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="trace-t03">T03</span> | [`test_values_against_reference`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="trace-t04">T04</span> | [`test_parameter_jacobian_against_symbolic_derivatives`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="trace-t05">T05</span> | [`test_low_velocity_and_inclination_limits`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="trace-t06">T06</span> | [`test_seeded_parameter_sweep`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="trace-t07">T07</span> | [`test_single_precision_parameter_sweep`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="trace-t08">T08</span> | [`test_newtonian_amplitude_reduction_and_polarization_limits`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="trace-t09">T09</span> | [`test_nonspinning_symmetries`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="trace-t10">T10</span> | [`test_compiled_waveform_against_reference`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="trace-t11">T11</span> | [`test_batch_and_jax_jit_against_scalar_reference`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="trace-t12">T12</span> | [`test_per_binary_frequency_grid_and_reference_vector`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |
| <span id="trace-t13">T13</span> | [`test_reference_frequency_neighbors`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py) |

<!-- END GENERATED TAYLORF2 TRACEABILITY -->

The drift checks in
[`tests/test_math_reference_traceability.py`](https://github.com/xangma/diffgw/blob/main/tests/test_math_reference_traceability.py)
require every canonical ID, complete pinned source metadata, resolvable
reference/frontend/test functions, and a current rendered map. These structural
checks keep the evidence navigable; the linked numerical tests establish the
declared comparisons, and source review establishes literature fidelity.

## Evidence and limitations

### Checks and numerical acceptance

The maintained numerical suite is
[`tests/test_taylorf2_math_reference.py`](https://github.com/xangma/diffgw/blob/main/tests/test_taylorf2_math_reference.py).
It compares each frontend directly to the reference, not just to the other
frontend. The public `diffgw.get_fd_waveform` dispatch path is checked for both
phase conventions with the cataloged nonspinning settings and cutoff disabled.
CPU FP64 values cover $(m_1,m_2)=(30,30),(1.4,1.3),(50,5)$ solar masses
at $f=20,31.25,80,160$ Hz, both phase modes, and face-on, inclined, and edge-on
orientations. First derivatives cover frequency, both component masses, distance,
inclination, time, phase, and reference frequency for the first two systems at
31.25 Hz, the asymmetric system at 8 Hz and edge-on inclination, and a face-off
neutron-star system one FP64 step above $f_{\rm ref}=40$ Hz.

Shared-frequency grids cover both $2\times3$ and $2\times2$ batch/frequency
shapes. Explicit per-binary grids and per-binary reference frequencies are also
checked. These tests exposed and corrected a broadcasting defect in both
Torch and JAX:
a shared one-dimensional frequency grid now gives every binary every frequency,
including when the two axis lengths happen to match. Explicit per-binary
frequencies use shape $(B,1)$. The mathematical specification did not change.
Compilation coverage uses JAX JIT and Torch `aot_eager` with `fullgraph=True`;
the Torch check covers Dynamo/AOT graph capture, not Inductor code generation.

| Quantity | Acceptance criterion |
| --- | --- |
| Each nonspinning PN coefficient | `rtol=2e-14`, `atol=2e-13` |
| Unwrapped phase, radians | `rtol=5e-14`, `atol=2e-10` |
| Positive amplitude | Relative error $\leq5\times10^{-14}$ |
| Each complex polarization | $|h-h_{\rm ref}|/A_{\rm ref}<3\times10^{-10}$ |
| First derivatives | `rtol=2e-9`, `atol=2e-9` after dividing amplitude/polarization rows by $A_{\rm ref}$; phase rows are unscaled |

The broader sweep uses seed `20260922` and 12 systems per precision and phase
mode. Total mass spans $[2.7,100]$ solar masses, mass ratio $[1,20]$, distance
$[10,2000]$ Mpc, inclination $[0,\pi]$, time shift $[-0.1,0.1]$ seconds, and
orbital phase $[-\pi,\pi]$. Mass, mass ratio and distance are sampled uniformly
in their logarithms. FP64 frequencies are chosen through $v\in[0.07,0.30]$
and $v_{\rm ref}\in[0.08,0.30]$. Additional cases test the 1.4/1.3 system at
1 Hz and the 50/5 system at 0.5 Hz, with face-on, edge-on, and face-off
inclinations, plus the exact reference frequency and its two adjacent FP64
values. Symmetry checks cover mass exchange, inverse-distance scaling and
inclination reflection.

These added value checks use a rounding budget based on the unreduced phase
scale, with $\epsilon$ the selected dtype's machine epsilon:

$$
K=|I(f)|+\mathbf{1}_{\rm reference}|I(f_{\rm ref})|
  +2\pi|ft_c|+2|\phi_c|+\pi.
$$

Phase absolute error must be at most $64\epsilon K$, amplitude relative error
at most $32\epsilon$, and polarization error $|h-h_{\rm ref}|/A_{\rm ref}$
at most $64\epsilon K+32\epsilon$. This is an engineering rounding budget,
not an interval proof; the tighter fixed-case gates above still run.
FP32 coverage is limited to $v,v_{\rm ref}\in[0.20,0.30]$ and additionally
requires the phase budget to be below **0.02 radians**. It does not qualify
long, low-frequency inspirals in FP32. The tests assert float32/complex64
outputs, with JAX x64 disabled for that sweep.

The CPU CI job installs both frontends and the reference dependencies, reports
their versions, and requires all collected mathematical-reference tests to run
without skips or expected failures. Its status applies to the revision and
environment being tested; rerun the commands below for a local checkout.

Amplitude normalization avoids an undefined relative error when a polarization
vanishes. These are regression gates on specified samples, not uniform error
bounds. Symbolic checks cover the Newtonian phase limit, phase gauge, and phase
anchor; domain and precision checks exercise the reference itself.

The [frozen fixture](https://github.com/xangma/diffgw/blob/main/tests/data/taylorf2_reference_v1.json)
stores 65 significant digits for four geometric-input cases in both modes,
including a near-reference frequency separated by $10^{-22}$ Hz. Values and
selected derivatives are evaluated at **100 and 160 decimal digits**, requiring
relative agreement within $10^{-65}$. The extra precision resolves cancellation
in near-reference derivatives. This fixture comes from the reference itself;
it detects subsequent drift but is not an independent validation of its equations.

## Reproduce the checks

From the repository root, using an environment with the optional dependencies:

```bash
python -m pip install -e '.[reference,torch,jax,dev]'
python -m diffgw_reference.artifacts --check
python -m diffgw_reference.traceability --check
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest tests/test_*math_reference*.py --require-math-reference --reference-device=cpu -ra
```

The normal optional-dependency tests can skip unavailable frontends; the
`--require-math-reference` mode makes missing dependencies, incomplete test
selection, skips, and expected failures fail the gate. An explicit CUDA run
uses `--reference-device=cuda` with compatible framework installations; the
maintained CI qualification is CPU. A passing CI job does not itself configure
repository branch protection.

Changes to equations, conventions, source versions, or constants need a source
and numerical impact review. Change the specification version and affected
component IDs when their mathematical meaning changes; update the map and
coverage in the same change. Renaming an implementation or test without changing
the mathematics needs a map update, not a new mathematical version. Record a
paper/code disagreement before adjudicating it, and preserve a documented model
choice rather than changing the reference to match a failing frontend.

After that review, regenerate the coefficient table and fixtures with
`python -m diffgw_reference.artifacts`, regenerate this map with
`python -m diffgw_reference.traceability`, and inspect both diffs. Neither
generated asset supplies an independent source derivation. To evaluate a case:

```python
from diffgw_reference import taylorf2 as tf2

result = tf2.evaluate_waveform(
    frequency="40", mass_seconds="0.0003", eta="0.24",
    distance_seconds="1e16", inclination="0.7", dps=80,
)
expr = tf2.derivative_expr(tf2.waveform_expr()["phase"], tf2.F)
```

Use decimal strings or exact SymPy numbers for deliberate high-precision inputs;
passing a Python float preserves its existing binary approximation. Evaluation
uses a private mpmath context and leaves the caller's precision unchanged.

The separately versioned aligned-spin, tidal, and spin-induced quadrupole
extensions have their own checks and limitations linked above. Eccentric,
higher-mode, and cutoff behavior remain outside this nonspinning specification.
Higher-order symbolic derivatives are available, but this first frontend
qualification covers the selected first derivatives above. No physical
calibration claim, formal proof, or whole-model verification follows from these checks.
