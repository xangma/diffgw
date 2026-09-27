# KerrRingdown direct-class physical-energy 220

Specification ID: `kerr-ringdown-220-energy-direct-class/v1`.

<!-- BEGIN GENERATED WAVEFORM DEFINITION -->

| Definition field | Catalog value |
| --- | --- |
| Model ID | `KerrRingdown220Direct` |
| Reference composition | `kerr-ringdown-220-energy-direct-class/v1` |
| Scope | Direct Torch/JAX KerrRingdown class only: one physical-energy spheroidal (2,2,0) mode with independent QNM; selected FP64 inputs. |
| Domain and modes | `time_frequency`; `(2, 2)` |
| Native backends | `torch`, `jax` |
| Scoped verification claims | 1 |

| Option | Default | Native choices | Reference-only choices |
| --- | --- | --- | --- |
| `call_route` | `direct_class` | `direct_class` | None |
| `energy_mode` | `(2,2,0)` | `(2,2,0)` | None |
| `angular_basis` | `spheroidal` | `spheroidal` | None |
| `qnm_backend` | `independent` | `independent` | None |

<!-- END GENERATED WAVEFORM DEFINITION -->

## Scope and status

This version covers **only** direct Torch/JAX `KerrRingdown.time_domain` and
`KerrRingdown.frequency_domain` class calls for one physical-energy
`(l,m,n)=(2,2,0)` mode. The required options are `energy_fraction>0`,
`energy_mode=(2,2,0)`, `amplitudes=None`, `phases=None`,
`angular_basis="spheroidal"`, and `qnm_backend="independent"`. The generated
QNM spline and spectral angular projection are checked against an independent
scalar equation solve at two selected spins. The `KerrRingdown` **dispatcher**,
other modes or options, and the separate opt-in `kerr220_td/fd` public-clean
functions are not included. The latter retain a distinct
[angular-convention blocker](kerr-ringdown-public-clean-gate.md).

## Inputs, conventions, and options

`final_mass` is the remnant mass in nominal solar masses, `distance` is in
Mpc, `energy_fraction=epsilon` is the energy fraction in the one mode,
`final_spin=a` is dimensionless, `inclination=theta` and `phi` are radians,
and `t0` is the onset in seconds. Define `M_s=final_mass*MSUN_SEC` and
`D_s=distance*MPC_SI/C_SI` from the [SI/IAU unit spec](units.md).
The QNM frequency `q=M_s*(omega-i*lambda)` has positive `omega` and
`lambda`. The spin weight is `-2`, angular dependence is `exp(+2i phi)`,
and the spheroidal function has unit sphere norm with positive real `(2,2)`
anchor. The causal step has `H(0)=1`; the FD convention is
`h_tilde(f)=integral h(t) exp(-2 pi i f t) dt` for signed frequency.

The direct class does not default to this slice: it defaults to a spherical
angular basis and a unit amplitude when `energy_fraction` is absent. All
listed options must therefore be explicit except the already-default
`qnm_backend="independent"` and `energy_mode=(2,2,0)`.

## Mathematics and components

The independent scalar 220 root solves the radial continued fraction and
angular eigenproblem of [Cook and Zalutskiy, Eqs. (4), (23)–(57)](https://arxiv.org/html/1410.7698v2).
The angular factor is a unit-normalized spectral expansion
`S(theta,phi;c)=sum_j b_j(c) {}_-2Y_j2(theta,phi)` at `c=a*q`.
The class projection is
`P=[S(theta,phi;c)+S(pi-theta,phi;c)]/2` and
`C=-i[S(theta,phi;c)-S(pi-theta,phi;c)]/2`.
The [angular specification](kerr-angular-projection.md) gives the Galerkin
operator and mirror symmetry; the [QNM specification](kerr-qnm-scalar-qualification.md)
gives the continued-fraction residual and branch.
For the same complex frequency and angular oblateness, the separated
Teukolsky ansatz in [Cook and Zalutskiy, Eq. (3)](https://arxiv.org/html/1410.7698v2)
has `exp(-i omega t) exp(+i m phi)`. The v1 direct class below instead has
`exp(+i omega t)`. Its scalar-reference agreement verifies the declared
composition, but does **not** show that this carrier and `S(a q)` form that
separated solution. The positive real spherical-coefficient anchor is also
distinct from an equatorial-real phase choice.

The finite-quality-factor **single-mode** energy coefficient is

$$A_1=\frac{\sqrt{32\pi\epsilon M_s/\lambda}}
{D_s\sqrt{1+(\omega/\lambda)^2}}.$$

It follows from the mode-energy flux in [Ruiz et al., Eq. (2.8)](https://arxiv.org/pdf/0707.4654v3)
with the unit angular norm of [Flanagan and Hughes, Eq. (3.22)](https://arxiv.org/abs/gr-qc/9701039).
The direct class uses the explicitly declared *native pair* coefficient
`A_pair=sqrt(2)*A_1`. For `u=t-t0>=0`, let
`z=A_pair exp[(-lambda+i omega)u]`; then

$$h_+(t)=\operatorname{Re}[zP],\qquad
h_\times(t)=\operatorname{Re}[zC].$$

Both are zero for `u<0`. Direct Fourier-minus integration gives, for either
complex projection `X=P,C`,

$$\widetilde h_X(f)=\frac{A_{\rm pair}}2 e^{-2\pi ift_0}
\left[\frac{X}{\lambda+i(2\pi f-\omega)}+
\frac{X^*}{\lambda+i(2\pi f+\omega)}\right].$$

These equations define the class-specific carrier. The conditional
[public-clean carrier](kerr-ringdown.md) uses a different sign and amplitude
convention, so its factors are not silently reused here.

## Implementation mapping

`diffgw_reference.kerr_ringdown.direct_class_220_inputs` obtains `q` from
`diffgw_reference.kerr_qnm.ScalarQNM220`, the angular projection from
`diffgw_reference.kerr_angular`, and the amplitude from the scalar energy
integral. Its `direct_class_220_time/frequency` functions compose the full
pair independently of either frontend. The Torch/JAX class methods use
project-generated QNM assets and their own spectral angular implementation.
The selected generated assets are frozen by SHA-256:

| Asset | SHA-256 |
| --- | --- |
| `src/torchwave/approximants/kerr_qnm_generated.py` | `e69d64e615dfd7bf054c2c7a9b721509688a1f126a482c7d9d0047f53b370dd3` |
| `src/diffgw_core/kerr_qnm_220_data.py` | `51b13ae3649ec332b45d4c538a669334ec31f4244105e3b4c2b0c24197f4c5e4` |
| `scripts/generate_kerr_qnm_independent.py` | `55f165af4a1c1106ba4a04f51c7099857210fddde36ca77eb03301dbd00e4ff9` |

These are project-generated numerical assets, not copied LAL constants.
`tests/test_kerr_ringdown_class_math_reference.py` is the maintained
whole-waveform and extrinsic-gradient gate;
`scripts/audit_kerr220_direct_class.py` emits per-backend errors and source
hashes for an independent run.

## Evidence and limitations

The frozen cases `(spin, mass/Msun, distance/Mpc, epsilon, theta, phi, t0/s)`
are `(0.35, 53, 230, 0.017, 1.1, 0.41, 0.002)` and
`(0.7, 92, 430, 0.005, 0.55, -0.27, -0.001)`. TD points are
`t0+(-0.001,0,0.002,0.008)` s and FD points are
`(-130,0,35,110,270)` Hz. The scalar run uses 40-digit mpmath,
radial depth 400, angular `ell_max=10` and Gauss–Legendre order 32;
`ell_max=12` changes the angular projection by at most `1.8e-14` here.
CPU FP64 whole-polarization relative L2 errors are below `5.2e-14` TD and
`3.4e-14` FD for both frontends on these points. The predeclared test gate
is `2e-10` relative L2, allowing numerical implementation variation while
remaining well above observed reference truncation.

Distance and positive-energy derivatives follow exactly from the amplitude:
`d h/d distance=-h/distance` and
`d h/d epsilon=h/(2 epsilon)` away from zero energy. Selected Torch/JAX
autodiff checks cover these extrinsic derivatives in TD and FD; angular
component tests cover selected angle derivatives separately. JAX's spectral
eigenvector uses `pure_callback`, so no JAX derivative through spin or
complex oblateness, nor a full intrinsic-gradient claim, is made.
The existing pinned public-binary
`KerrRingdown_holdout_off_axis_energy_onset` exercises the **dispatcher**, which
selects `kerr220_fd` when no explicit `energy_mode` is passed. It is not
binary evidence for this direct-class slice. A staged diagnostic compares
`(53 Msun, spin 0.35 or 0.7, 230 Mpc, epsilon 0.017, theta 1.1, phi 0.41,
t0=0)` against LALSuite `6.2.1` `SimBlackHoleRingdown` public output. The
public time series is transformed by the repository's converged Simpson
adapter on a 1 Hz grid over 50–500 Hz. Refinement from 65536 to 131072 Hz
changes it by at most `1.26e-8` relative L2.

| Diagnostic waveform; plus/cross relative L2 | Spin 0.35 | Spin 0.7 |
| --- | --- | --- |
| Direct class v1 as declared | `1.232 / 1.554` | `1.238 / 1.536` |
| Direct class with empirical `phi -> -phi`, intrinsic phase `pi` | `5.05e-4 / 1.30e-2` | `1.01e-3 / 2.05e-2` |
| Independent scalar spectral angular factor, paper's minus carrier, equatorial-real phase | `1.92e-4 / 1.99e-4` | `3.65e-4 / 3.83e-4` |
| Separate `kerr220_fd` inferred-table route | `5.23e-8 / 5.85e-8` | `1.00e-7 / 9.03e-8` |

The phase map is an empirical binary diagnostic, not a sourced definition
of the public API's phase input. The paper-carrier row uses the independent
scalar QNM, spectral angular solution and energy integral; its equatorial
phase is `arg S(pi/2,0)`, computed without the binary. Even after that
choice, the independent spectral angular values differ from the separately
inferred table shape at the inclination and mirror by `1.87e-4 / 3.70e-4`
for spin 0.35 and `3.51e-4 / 9.12e-4` for spin 0.7. The table row is a
different conditional route and cannot qualify the independent direct class.
This v1 definition remains **sampled implementation verification only**:
the paper-carrier and angular-profile results explain much of the mismatch,
but do not establish public-binary compatibility or physical accuracy.

## Reproduce the checks

Run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests JAX_ENABLE_X64=1
JAX_PLATFORMS=cpu python -m pytest -q
tests/test_kerr_ringdown_class_math_reference.py` in an environment with
Torch, JAX, SymPy and mpmath. For a JSON receipt, run
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src JAX_ENABLE_X64=1
JAX_PLATFORMS=cpu python scripts/audit_kerr220_direct_class.py`.
With LALSuite installed, the diagnostic binary comparison is reproducible
with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests python
scripts/audit_kerr220_direct_class_binary.py` (and `--spin 0.7` for the
second column). This audit reads public binary waveform output only; it
does not inspect LAL implementation or fit the independent scalar model.
