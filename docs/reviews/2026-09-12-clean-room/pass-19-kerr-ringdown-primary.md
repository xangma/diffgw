# Pass 19: primary-source review of the Kerr ringdown waveform

Decision: **retain the current native primitive unchanged, but reject any claim
that it is a source-complete Kerr ringdown waveform**. The implementation has a
well-defined real, causal, equal-twin-mode convention and its Fourier expression
is exact for that convention. The primary sources leave the excitation
coefficients to the initial data, while the current API supplies one amplitude
and phase per positive-𝑚 mode. The missing twin-mode data, physical distance
normalization, and default angular convention cannot be filled in without
changing native semantics or guessing metadata.

## Primary sources and hashes

Only original papers were used for the derivation. No external implementation
source, translated port, binary, verification result, residual, notebook, or git
history was consulted.

| Source | Use | SHA-256 |
| --- | --- | --- |
| [Berti, Cardoso & Will, gr-qc/0512160v2](https://arxiv.org/pdf/gr-qc/0512160v2) | Eqs. (2.4)--(2.9), redshift/distance prescription, Eqs. (3.13)--(3.19) | `5fe24e98616fee024a1e4e577129ff655480f733f451a7c06cc9c1213352e525` |
| [Berti & Cardoso, gr-qc/0605118v2](https://arxiv.org/pdf/gr-qc/0605118v2) | QNM excitation factors and initial-data-dependent coefficients, Eqs. (3.7)--(3.9) | `bf3a441889fad473a2ae57842984a936e9c541af90c812c197d9245a989b5242` |
| [Cook & Zalutskiy, 1410.7698v2](https://arxiv.org/pdf/1410.7698v2) | Spin-weighted spheroidal normalization and ±m/frequency symmetries, Eqs. (47)--(49), (57)--(58) | `33973382059461835989057a47e93336e5233cccede48529b4b19cdc306d7330` |

The inspected implementation and direct native test snapshots were:

```text
src/torchwave/approximants/kerr_ringdown.py  dfe4fcb4bd90e01cb2038a97863bccd747468a8d1b196b4c6c668784c9442a4b
tests/test_ringdown.py                       b8a3888a5b0cef3d0af1b1052bf01b02d557080b5f28ca4443ced596b1783b72
```

## Source-complete target

Let (u=t-t_0), (gamma_{ell mn}=1/\tau_{ell mn}>0), and use the
Berti--Cardoso--Will convention (e^{+i\omega t}). Their ringdown ansatz is

```text
h_+ + i h_x = (M/D_L) sum_{l,m,n} A_lmn exp[i(omega_lmn u + phi_lmn)]
               exp[-gamma_lmn u] S_lmn(theta,phi),   u >= 0.
```

The mode frequencies are complex QNM eigenvalues; (n=0) is the longest-lived
mode and successive (n) label faster damping. The dimensional quantities are
in geometric units, with (M\omega) and (M\gamma) dimensionless. For a
cosmological source, the same paper prescribes luminosity distance and
detector-frame (redshifted) mass, frequency, and damping time together.

For a unit-normalized angular function, direct integration of the asymptotic
quadratic flux gives, for one isolated complex coefficient,

```text
epsilon = A^2 * (omega_hat^2 + gamma_hat^2) / (32*pi*gamma_hat),
A^2 = 32*pi*epsilon*gamma_hat / (omega_hat^2 + gamma_hat^2),
```

where hats denote multiplication by the remnant geometric mass and
(\epsilon=E/(M c^2)). This is the same finite-(Q) normalization as BCW
Eq. (3.18), after (Q=\omega/(2\gamma)) and (f=\omega/(2\pi)). It is a
normalization of a supplied coefficient; it does not predict (A).

The angular target is the spin-(-2) spheroidal eigenfunction evaluated at the
complex QNM oblateness (c=a\omega), with unit sphere norm. Cook--Zalutskiy
Eq. (47) fixes the azimuthal factor and their spectral construction fixes the
phase continuously from the spherical function at (c=0). The (m\to-m)
relation is not an optional numerical detail: BCW Eq. (2.7), and
Cook--Zalutskiy Eqs. (48b)--(48c), pair the opposite-(m) angular solution with
the opposite-frequency QNM.

BCW Eq. (2.9) then requires two independent real amplitudes and phases for each
positive-(m) label. Equal twin coefficients are only a restricted polarization
choice. The source explicitly notes that pure plus and pure cross states are
special relations between the two coefficients; a single positive-frequency
coefficient cannot represent arbitrary polarization.

The excitation paper makes the remaining underdetermination explicit:

```text
C_n = B_n * integral(initial-data source at omega_n),
```

so QNM frequencies and universal excitation factors do not determine the mode
amplitudes, phases, or overtone weights. No primary source justifies selecting
only (n=0), or any fixed set of overtones, for a generic merger.

## Comparison with the native API

The current time-domain path evaluates, for each supplied positive-(m) mode,

```text
s = amplitude * exp[-u/tau + i*(omega*u + phase)],
d_plus  = [S_minus + (-1)^l conj(S_plus)] / 2,
d_cross = i*[S_minus - (-1)^l conj(S_plus)] / 2,
h_plus  = Re(s*d_plus),
h_cross = Re(s*d_cross),
```

with a causal (u\ge0) gate. This is internally a real equal-twin-mode
convention. It is a valid restricted member of the BCW family, but it does not
implement the four independent twin-mode constants in Eq. (2.9). The API also
does not accept a source distance or multiply the result by (M/D_L); its
amplitude is therefore a native modal input, not a complete physical strain
normalization. Adding distance or silently converting the amplitude would
change the established native meaning.

The Fourier path uses

```text
H(f) = integral h(t) exp(-2*pi*i*f*t) dt
```

and both poles of each real causal exponential. This agrees algebraically with
the stated time convention: each pole is

```text
1 / [gamma + i*(2*pi*f - omega)]
```

and its conjugate negative-frequency partner, with the onset factor
`exp(-2*pi*i*f*t0)`. The existing Hermitian and time-shift behavior is therefore
source-consistent for the native restricted waveform.

The default `angular_basis="spherical"` is another deliberate approximation:
the source-complete Kerr target requires the complex spheroidal function at
`c = (chi_f*M)*(omega - i/tau)`. The opt-in `angular_basis="spheroidal"` path
does evaluate that equation-defined factor. Changing the default would alter
prior native outputs; the sources do not provide excitation metadata that would
justify compensating for that change.

Mass units are otherwise consistent for the QNM quantities: the implementation
converts solar masses to seconds, returns `omega` in rad/s and `tau` in seconds,
and forms dimensionless `M*(omega-i/tau)` for the spheroidal argument. The
published (M/D_L) strain factor remains an explicit API gap.

## Conclusion and changes

No runtime or test change was made. A source-complete correction would require
at least a documented twin-mode amplitude/phase interface, an explicit distance
and strain convention, and a declared overtone excitation model. The primary
papers determine none of those merger-dependent inputs. The smallest justified
action is therefore to retain the native equal-twin, amplitude-supplied model
and document its limits rather than guess them.

## Validation

The focused direct/native checks were run without any frozen public-binary
verification:

```text
python -m pytest -q tests/test_clean_room_ringdown.py \
  tests/test_clean_room_ringdown_energy.py tests/test_ringdown.py
60 passed, 1 skipped
```

These tests cover the causal time/Fourier pair, Hermitian symmetry, time shift,
spherical/spheroidal paths, mass/spin/inclination/start-time broadcasting, and
autograd. They validate the current convention; they do not establish a unique
merger excitation model or the omitted physical distance factor.
