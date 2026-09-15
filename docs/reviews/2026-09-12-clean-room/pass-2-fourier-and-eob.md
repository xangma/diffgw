# Ringdown transforms and EOB waveform normalization

This implementation pass used direct Fourier derivation, fresh primary papers,
and the inherited native modules. LAL was used separately through its public
binary API for verification. No LAL implementation source, reference-data
coefficients, or fitted waveform residuals entered these changes.

## Causal ringdown Fourier pair

For a real polarization `h(t)=Re[C exp((-gamma+i omega)(t-t0))] Theta(t-t0)`
and Fourier kernel `exp(-2 pi i f t)`, direct integration gives

```
h(f) = exp(-2 pi i f t0)/2 * [
    C/(gamma + i(2 pi f - omega))
  + conj(C)/(gamma + i(2 pi f + omega))].
```

The earlier transform omitted one pole, introduced an extra `1/(2 pi)`
normalization, and used the opposite pole sign. The native TD and FD
polarizations now form this Fourier pair. Before onset, the implementation
clamps elapsed time before exponentiation, preventing overflow followed by
`Inf*0`. The real two-pole spectrum's maximum need not be exactly `omega/(2 pi)`.

Tests use independent numerical time-domain quadrature over positive and
negative frequencies for three modes, Hermitian symmetry, delay covariance,
finite pre-onset derivatives, and complex gradient checks. Together with the
existing ringdown tests, **10 passed** in the initial focused run.

These changes retain the inherited QNM fit and spherical angular projection.
[Berti et al., gr-qc/0512160v2](https://arxiv.org/pdf/gr-qc/0512160v2)
distinguishes numerical Kerr spectra, fitted mode frequencies, and angular
functions. The native fitted/spherical model does not thereby become identical
to a numerical spheroidal LAL ringdown. Full independent-reference coverage
remains unresolved.

## EOB projection, torque, and physical scaling

The leading mode is `h22=-8 sqrt(pi/5) nu v^2 exp(-2i phi)`. Pairing it with
`h2,-2=conj(h22)` gives real polarizations

```
hplus  = sqrt(5/(16 pi)) (1+cos(iota)^2) Re(h22)
hcross = -sqrt(5/(4 pi)) cos(iota) Im(h22).
```

These follow from the mode expansion and spherical harmonics in
[García-Quirós et al., Eqs. (2.1)-(2.5)](https://arxiv.org/pdf/2001.10914v2)
and the Newtonian mode in
[Damour, Iyer, Nagar](https://arxiv.org/pdf/0811.2069v2).
Transforming the real polarizations retains both Fourier components. Taking
the positive bins of the negative-frequency complex mode had lost the carrier.
The orbital phase now rotates the TD mode before projection. A positive time
delay multiplies the spectrum by `exp(-2 pi i f tc)`.

The Fourier measure retains its differentiable `dt` in seconds. The remnant
frequencies returned by PhenomD are dimensionless `M*f`; ringdown is therefore
constructed in `t/M`. Those units were already consistent in the inherited
code. The repaired ringdown sample grid starts at the actual time step.
The requested lower frequency reaches the dynamics explicitly, and the
Hamiltonian supplies the angular velocity used in the leading waveform.
Broadcast parameter and frequency grids preserve their full batch prefix.

For momentum normalized by reduced mass, the torque is
`F_phi=-Flux/(nu Omega)`, as in
[Damour–Nagar, Eq. (12)](https://arxiv.org/pdf/0704.3550v2).
The omitted division by `nu` is restored. A stable signed denominator keeps
the radial force finite at zero angular momentum.

The focused waveform/torque suite has **10 passing tests**: exact periodic
Fourier amplitudes, finite-signal phase rotation, delay covariance, mass
similarity, finite-difference gradients, scalar/batch equivalence, and energy
balance. A separate agent independently reviewed the projection and torque;
its phase and broadcasting findings were repaired and covered by these tests.

## External EOB reference adapters

The canonical native ODE and TEOB cases now call LAL `SimInspiralFD`, which
transforms its TD model; the ROM case calls `SimInspiralChooseFDWaveform`.
The ODE reference is generated through 1024 Hz for its Nyquist requirement,
while comparison remains below the original 256 Hz bound. A finite TD Fourier
transform can have nonzero reference DC. This is retained, validated for
finiteness, and excluded by the declared positive comparison band; native DC
and support checks and all in-band tolerances remain enforced.

The user approved downloading the LAL ROM reference dependency. Its origin,
size and checksums are recorded in [oracle-data.json](verification/pass-2/oracle-data.json).
It is loaded only by LAL. TorchWave does not read it, and it is not native ROM
training data. The fetch script and CI cache validate its published checksum.

The [initial external measurements](verification/pass-2/eob.json) fail for all
three EOB models. These expose previously hidden model differences; the former
native self-comparisons supplied no independent evidence of LAL agreement.
See [conservative repair and remaining physics](pass-2-eob-conservative.md).
