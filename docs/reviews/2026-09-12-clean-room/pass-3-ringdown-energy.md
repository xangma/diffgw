# Ringdown energy-fraction normalization from primary equations

For an isolated complex mode with unit angular norm, the exact derivative-energy
integral gives

```text
H = h_plus - i*h_cross = a * exp[-(gamma+i*omega)*u] * S(theta,phi), u>=0
integral_sphere |S|^2 dOmega = 1
E_single = c^3*D^2*|a|^2*(omega^2+gamma^2)/(32*pi*G*gamma).
```

Here `a` is the complex **strain modal coefficient at onset**, before angular
projection; `u=t-t0`, `gamma=1/tau>0`, `omega>0`, and `D` is the wave-zone
distance. This is exact for the stated exponential and asymptotic quadratic
flux. It does not determine how strongly a merger excites that mode.

An equal-amplitude equatorial `+m,-m` pair carries twice this energy for `m>0`.
The current native API's amplitude is twice the coefficient of either member
of its pair. These three amplitudes therefore require distinct conversions.
This document derives the conversions; it makes no implementation change.

## Primary sources and provenance

PDFs were freshly retrieved on 2026-09-12. Equation numbers refer to these
versions. No external waveform code, binary-oracle output, fitting, or previous
calibration document was used.

| Primary PDF | Equations used | SHA-256 |
| --- | --- | --- |
| [Ruiz, Alcubierre, Nunez and Takahashi, 0707.4654v3](https://arxiv.org/pdf/0707.4654v3) | 2.4–2.8: polarizations, complex strain and sphere-integrated energy flux | `d92ad1c0011daa5fadabafcc3a19aa4bc7a4e855e9f3043b6e0d1101e4d3858d` |
| [Flanagan and Hughes, gr-qc/9701039v2](https://arxiv.org/pdf/gr-qc/9701039v2) | 2.41; 3.21–3.22: damped complex strain and angular norm; 3.25–3.26: high-Q energy; Appendix B: onset/extension distinction | `c7c2f7b8184b800e1f19422698b9e22d5f61e5c910220ec5cb5e776cf51be5ed` |
| [Berti, Cardoso and Will, gr-qc/0512160v2](https://arxiv.org/pdf/gr-qc/0512160v2) | 3.1; 3.15–3.19: efficiency and finite-Q amplitude; discussion before 3.5: FH extension versus one-sided transform | `5fe24e98616fee024a1e4e577129ff655480f733f451a7c06cc9c1213352e525` |
| [Cook and Zalutskiy, 1410.7698v2](https://arxiv.org/pdf/1410.7698v2) | 47–49, text after 56: angular symmetries and unit coefficient norm; 57: mirror-frequency branch | `33973382059461835989057a47e93336e5233cccede48529b4b19cdc306d7330` |

Local PDFs and text extracts are in `/tmp/torchwave-clean-ringdown/`.

## Derivation and SI conversion

Ruiz 2.8 gives, in geometric units,

```text
dE/du = D^2/(16*pi) * integral_sphere |dH/du|^2 dOmega.
```

Their text explicitly removes the cycle average for the sphere/time-integrated
expression. Inserting the mode above gives `|dH/du|^2 =
|a|^2*(omega^2+gamma^2)*exp(-2*gamma*u)*|S|^2`. Integrating over the sphere
and `0<u<infinity` supplies factors `1` and `1/(2*gamma)`. Restoring units
multiplies the flux prefactor by `c^3/G`, yielding the opening result.

Define an efficiency with respect to the **remnant** rest energy and dimensionless
QNM frequencies as follows:

```text
epsilon = E_mode/(M_f*c^2)
M_time = G*M_f/c^3
omega_hat = M_time*omega
gamma_hat = M_time*gamma
Q = omega/(2*gamma)
a = [G*M_f/(c^2*D)] * A * exp(i*phi0).
```

For one isolated mode,

```text
epsilon = A^2*(omega_hat^2+gamma_hat^2)/(32*pi*gamma_hat)
A_exact = sqrt[32*pi*epsilon*gamma_hat/(omega_hat^2+gamma_hat^2)]
        = sqrt[16*pi*epsilon/(omega_hat*Q*(1+1/(4*Q^2)))].
```

Equivalently, the SI strain amplitude is
`|a|=sqrt[32*pi*G*epsilon*M_f*gamma/(c*D^2*(omega^2+gamma^2))]`.
Mass must be expressed in kg in this SI expression. If efficiency instead
uses a different reference mass `M_ref`, replace `epsilon` in the dimensionless
amplitude formula by `epsilon*M_ref/M_f`.

The displayed SI flux concerns a nearby source. For cosmological strain,
use luminosity distance and detector-frame mass/frequencies together, as
specified in BCW immediately before Sec. III. The dimensionless amplitude
and source-frame efficiency remain the same; a local flux integral using
luminosity distance is not itself the source-frame emitted energy.

## Finite Q and the published high-Q approximation

Dropping the damping contribution to the derivative gives

```text
E_highQ = E_exact / (1+1/(4*Q^2))       [fixed amplitude]
A_highQ = sqrt[16*pi*epsilon/(omega_hat*Q)]
A_exact/A_highQ = (1+1/(4*Q^2))^(-1/2) [fixed efficiency].
```

This ratio is `0.70710678`, `0.89442719`, `0.97014250`, and `0.99875234`
at `Q=0.5,1,2,10`, respectively. No assumption of large Q is required by the
exact integral. The approximation is less accurate for strongly damped modes.

FH 3.26 gives `E ~= A^2*M_f^2*f*Q/8`, with `f=omega/(2*pi)`, precisely
the high-Q expression above. BCW 3.18 gives
`A^2=32*Q*epsilon/[M_f*f*(1+4*Q^2)]` in geometric units. Substituting
`M_f*f=omega_hat/(2*pi)` reproduces `A_exact` exactly. Although BCW obtains
this through the FH extended spectrum, that scalar normalization agrees with
the direct complex derivative-energy integral. This agreement does not equate
the FH spectrum with the exact Fourier transform of a hard-start waveform.

## Complex phase, real polarizations and onset

Let `S=|S|*exp(i*sigma)`, so `phi0=arg(a)` and the local phase in the real
polarizations is `theta=omega*u-phi0-sigma`. Then

```text
h_plus  = |a*S| * exp(-gamma*u) * cos(theta)
h_cross = |a*S| * exp(-gamma*u) * sin(theta).
```

At onset the complex phase is `phi0`; the local complex strain additionally
contains `sigma`. Neither phase changes the isolated-mode energy because the
sum of the two squared polarization derivatives is `|dH/du|^2` pointwise.
There is no extra factor of two for having two real polarizations.

For clarity, put `theta=omega*u+theta0` and remove the common squared
amplitude. The two derivative integrals separately are

```text
I_plus  = (omega^2+gamma^2)/(4*gamma)
          + [gamma*cos(2*theta0)+omega*sin(2*theta0)]/4
I_cross = (omega^2+gamma^2)/(4*gamma)
          - [gamma*cos(2*theta0)+omega*sin(2*theta0)]/4.
```

Their sum is phase independent. Normalizing a lone real cosine, one local
polarization, or a detector-projected signal therefore requires its own
angular/polarization convention; its amplitude cannot simply replace `a`.

Energy here means the integral over the physical ringdown segment `u>0`.
A discontinuous extension `Theta(u)*H(u)` has an onset delta function in its
derivative; squaring it does not define a finite physical energy. The formula
excludes that artificial jump. FH Appendix B discusses the spectral consequences
of choosing an extension before onset. A smooth physical attachment can carry
additional energy and must be specified to include it. Moving the start later
along the same exponential by `Delta` changes its remaining energy by
`exp(-2*gamma*Delta)`.

## Equatorial mirror pair and current native amplitude parameter

For spin weight `-2`, use unit-normalized angular functions

```text
S_plus  = S_(ell,m)(theta,phi; c), c=chi_f*(omega_hat-i*gamma_hat)
S_minus = S_(ell,-m)(theta,phi; -conj(c))
        = (-1)^ell * conj[S_plus(pi-theta,phi; c)]
H_pair  = a(u)*S_plus + (-1)^ell*conj[a(u)]*S_minus.
```

The angular relation follows from Cook–Zalutskiy 48b–c, with the mirror
frequency from 57. For `m>0`, opposite azimuthal factors give
`integral S_plus*conj(S_minus) dOmega=0`. Both members have the same damping
and equal coefficient magnitude, so `E_pair=2*E_single`. This does not require
different spheroidal eigenfunctions at the same m to be orthogonal. For `m=0`
there is no such azimuthal orthogonality argument and this doubling rule must
not be applied automatically.

Read-only inspection of the current native `KerrRingdown.time_domain` finds

```text
s = amplitude_input * exp(-gamma*u+i*(omega*u+phase_input))
d_plus  = (S_minus+(-1)^ell*conj(S_plus))/2
d_cross = i*(S_minus-(-1)^ell*conj(S_plus))/2
h_plus  = Re(s*d_plus), h_cross = Re(s*d_cross).
```

Algebra therefore gives

```text
H_native = [s*S_minus+(-1)^ell*conj(s)*S_plus]/2
a_native(u) = (-1)^ell*conj(s)/2
|a_native(0)| = |amplitude_input|/2
phi0 = pi*ell-phase_input  (mod 2*pi), for positive amplitude_input.
```

Thus the current projection already builds a mirror pair, with half the input
amplitude in each complex coefficient. Define
`B=32*pi*epsilon*gamma_hat/(omega_hat^2+gamma_hat^2)`. At a fixed total
efficiency the required quantities are:

| Quantity to supply | Dimensionless magnitude; multiply by `G*M_f/(c^2*D)` for strain |
| --- | --- |
| Isolated single complex-mode coefficient | `sqrt(B)` |
| Each coefficient of an equal-amplitude physical mirror pair | `sqrt(B/2)` |
| Current native `amplitude_input`, whose pair members have half its magnitude | `sqrt(2*B)` |

The current native amplitude is consequently `sqrt(2)` times the isolated
single-mode coefficient for the same total energy, or twice the per-member
coefficient for a pair. No additional inclination or azimuth factor belongs
in this energy conversion: the angular functions perform that projection.

Inspection snapshot SHA-256 values, to keep this API observation distinct
from the primary physical derivation:

- `kerr_ringdown.py`: `6f0fa414ef790cdc48e0cac85d0dc3ce78239586952692d83c302b3307cd1d33`
- `kerr_angular.py`: `68e1741a140c79b69fe83993c378b98c89137bf4d9e4c6fd80ed10d403cc9e9c`
- `imrphenom_hm.py` (spherical projection helper): `cfe44564b1f5ca8546f3194ee0ca1166f96375945f90857c5e7aa99115ce4e42`

## Scope and independent algebra checks

For several coherent modes, isolated energies need not add: overtones with
the same m can interfere. With `lambda_j=gamma_j+i*omega_j` and
`G_ij=integral S_i*conj(S_j) dOmega`, the geometric-unit integral is

```text
E = D^2/(16*pi) * sum_ij a_i*conj(a_j)*lambda_i*conj(lambda_j)*G_ij
                             /(lambda_i+conj(lambda_j)).
```

This reduces to the results above for a single mode or an orthogonal mirror
pair. A future multi-overtone efficiency API must state whether it means
isolated component energies or the total coherent energy.

Direct numerical quadrature of the two real derivative integrals at
`(gamma,omega,theta0)=(1,1,0),(.2,1.7,.37),(1.3,.4,-1.2)` agreed with the
displayed formulas within `4e-15`; their summed relative errors were below
`3e-16`. Independent sphere quadrature of the analytic spherical 22 functions
with 32 Gauss–Legendre and 64 azimuth points gave angular-energy weights
`1`, `2`, and `1/2` for a single mode, a physical pair, and the current native
input convention, respectively, within `8e-15`, at three initial phases.
These check the derivation and projection algebra without waveform-oracle
values. No implementation or tests were changed for this derivation task.

## Independent derivation review

A separate review checked the four primary PDFs against the SHA-256 values
above and read the native projection functions directly. The finite-Q factor
`omega^2+gamma^2`, the `m>0` pair-energy factor of two, and the conversion
factors `1`, `1/sqrt(2)`, `sqrt(2)` follow with the stated unit angular norm.
Expanding the native real projections recovers `a_native=(-1)^ell*conj(s)/2`
and hence `phi0=pi*ell-phase_input` for positive input amplitude. The onset
convention consistently integrates the smooth segment after `t0`, excludes
the distributional jump, and rescales remaining energy by the squared decay
when onset is moved along the same exponential. No defect was found in these
claims; the restrictions concerning `m=0` and coherent overtones remain needed.

Additional independent 48-point Gauss–Legendre by 64-point azimuth quadrature
checked native complex spheroidal modes `(ell,m)=(2,2),(3,2),(4,3)` at
`c=0.3-0.05i,0.5-0.15i,0.8-0.2i`, including even and odd ell and three
initial phases. Angular norms differed from one by at most `9.8e-15`,
mirror overlaps were below `2.8e-16`, the native pair reconstruction error
was below `1.2e-16`, and its derivative-energy weight `1/2` agreed within
`1.0e-14` relative error. The three real-polarization temporal integrals
were separately reproduced within `4.0e-15` absolute error. These are
primary-equation and native-algebra checks, with no external waveform
implementation, oracle, or calibration input.


## Standalone implementation record — 2026-09-12

The subsequent implementation adds
`src/torchwave/approximants/kerr_ringdown_energy.py` and
`tests/test_clean_room_ringdown_energy.py`. Existing waveform generators and
their defaults are unchanged. The function signature is:

```python
ringdown_amplitude_from_energy_fraction(
    final_mass, distance, energy_fraction, omega, tau,
    *, convention="native_pair",
)
```

Inputs are remnant solar masses, Mpc, the energy fraction relative to remnant
rest energy, rad/s, and seconds, respectively. The output is a nonnegative
onset **strain amplitude**. `single_mode`, `pair_member`, and `native_pair`
implement the three conventions derived above. For both pair conventions,
the fraction denotes the total energy of an equal-amplitude positive-m/mirror
pair; no m=0 or coherent-overtone energy prescription is implied. Frequency
and damping time are caller inputs; the module imports no QNM backend,
waveform tables, oracle data, or other approximant implementation.

The implementation evaluates the exact finite-Q expression in logarithms to
avoid intermediate overflow/underflow from SI scales and large frequency-time
products. The energy square root remains a separate factor: at zero energy,
the amplitude and derivatives with respect to positive physical parameters
are zero, while the right derivative with respect to energy fraction is
infinite. Negative/nonfinite energy and nonpositive/nonfinite physical
parameters raise errors. Complex and boolean inputs and floating tensor
precisions other than float32/float64 are rejected.

Parameters broadcast; floating tensor dtypes promote between float32/float64,
with float64 for Python-only or integer-only input. Tensor inputs must share
a device, and conversions preserve autograd. No high-Q approximation or
inclination factor is silently applied.

Focused validation:

```text
python -m pytest -q tests/test_clean_room_ringdown_energy.py
43 passed, 1 skipped
```

The skip is the conditional CUDA device/gradient test on this host. CPU
float32/float64, mixed precision promotion, inference mode, and all five
parameter gradients were exercised. Gradcheck scales physical strains by
1e21 to avoid a vacuous absolute tolerance, and separate analytic logarithmic
derivatives check every input. Zero-energy derivatives, extreme finite
inputs, invalid domains, broadcasting, and dimensional mass/distance/energy
scalings are covered.

Independent energy tests integrate the squared real polarization derivatives
with SciPy over `u/tau` from zero to infinity, using Gauss–Legendre polar and
uniform azimuth quadrature. Explicit spherical 22 functions cover all three
conventions at Q=0.5,2,10. The native projection is additionally integrated
for spherical 21 and 33 and numerical spheroidal 32 with complex
spheroidicity, at three initial phases. The recovered energy fractions agree
within a 5e-13 relative tolerance. These checks use caller-selected frequency
and damping values and do not access a frequency backend or stored waveform
oracle. A separate direct SI calculation and the published finite-Q/high-Q
ratio verify the unit conversion and damping-derivative contribution.
