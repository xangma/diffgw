# Pass 20: source-defined physical Kerr ringdown subset

Decision: add an opt-in physical normalization path for one positive-
`m` mode and its equal-amplitude equatorial mirror. Keep the existing
amplitude-supplied native path and spherical default unchanged. Generic twin
amplitudes, coherent multi-mode energy, `m=0` pair doubling, and merger
excitation amplitudes remain explicit API gaps.

## Clean-room provenance

This pass used only primary papers and the independent derivations already
recorded in
[`pass-3-ringdown-energy.md`](pass-3-ringdown-energy.md) and
[`pass-19-kerr-ringdown-primary.md`](pass-19-kerr-ringdown-primary.md). The
primary equations are from [Ruiz et al., 0707.4654v3](https://arxiv.org/pdf/0707.4654v3),
[Flanagan and Hughes, gr-qc/9701039v2](https://arxiv.org/pdf/gr-qc/9701039v2),
[Berti, Cardoso and Will, gr-qc/0512160v2](https://arxiv.org/pdf/gr-qc/0512160v2),
and [Cook and Zalutskiy, 1410.7698v2](https://arxiv.org/pdf/1410.7698v2).
No LAL or PyCBC source was inspected, and no public binary output was used to
fit a runtime constant.

## Fixed source convention

For a unit-normalized angular function and `u=t-t0 >= 0`, write

```text
H = h_plus - i h_cross = a exp[-(gamma+i omega)u] S(theta,phi)
E_single = c^3 D^2 |a|^2 (omega^2+gamma^2)/(32*pi*G*gamma).
```

With `epsilon=E/(M_f c^2)`, `M_time=G M_f/c^3`, and hats denoting
dimensionless quantities, the exact finite-Q coefficient is

```text
A_single = sqrt(32*pi*epsilon*gamma_hat /
                 (omega_hat^2 + gamma_hat^2))
|a| = sqrt(32*pi*G*epsilon*M_f*gamma /
            (c*D^2*(omega^2+gamma^2))).
```

The implementation accepts `final_mass` in solar masses, `distance` in Mpc,
physical `omega` in rad/s, and `tau=1/gamma` in seconds. The existing helper
`ringdown_amplitude_from_energy_fraction` performs the SI conversion without
introducing an inclination or azimuth factor: angular projection performs that
operation later.

For `m>0`, Cook--Zalutskiy gives the equatorial partner

```text
S_minus(theta,phi;-conj(c)) = (-1)^l conj(S_plus(pi-theta,phi;c)),
c = (M_f chi_f) (omega - i/tau).
```

The two azimuthal branches are orthogonal on the sphere, so an equal-
amplitude pair carries twice the isolated coefficient energy. The native
projection puts half the supplied input amplitude in each member. Therefore
the physical total-pair conversions are:

```text
single_mode: sqrt(B)
pair_member: sqrt(B/2)
native_pair: sqrt(2*B),
B = 32*pi*epsilon*gamma_hat/(omega_hat^2+gamma_hat^2).
```

`native_pair` is the value needed by the current equal-twin projection.
Fourier behavior remains the exact transform of the causal real waveform,
using `integral h(t) exp(-2*pi*i*f*t) dt`, both frequency poles, and the onset
factor `exp(-2*pi*i*f*t0)`.

## Runtime scope

`KerrRingdown.time_domain`, `KerrRingdown.frequency_domain`, and the public
`get_ringdown_waveform` now accept the opt-in tuple
`(distance, energy_fraction, energy_mode)`. It is source-complete only when
`angular_basis="spheroidal"`, `energy_mode=(l,m,n)` has `l>=2`, `0<m<=l`,
and `n>=0`. The path computes the QNM frequency with the selected backend and
derives one native equal-mirror amplitude from the total energy fraction.

The runtime rejects missing distance, spherical projection on this physical
path, simultaneous explicit amplitudes and energy, and `m=0`. These guards
keep underdetermined conventions visible. Explicit amplitudes still preserve
the prior native semantics, and the spherical default is unchanged.

The papers do not determine independent positive/negative-`m` excitation
coefficients, generic merger excitation phases, or the coherent energy of
several modes. Those cases remain unsupported rather than being inferred from
the isolated-mode formula.

## Validation

The focused tests in
[`test_clean_room_ringdown_physical.py`](../../../../tests/test_clean_room_ringdown_physical.py)
check exact equality between the derived-energy and explicit-native paths,
inverse-distance scaling through the public waveform API, and all ambiguity
guards. Together with the existing native and energy tests:

```text
63 passed, 1 skipped
```

`git diff --check` is clean. No LAL/PyCBC implementation source or fitted
oracle constants were used.
