# NSBH conditional complex polarization composition, v1

## Scope and status

Specification **`nsbh.polarization-composition/v1`** checks the positive-frequency
three-window amplitude, an independently supplied explicit-gauge phase, and
the aligned quadrupole projection into both complex polarizations. This is a
conditional executable gate, not qualification of the complete public NSBH
waveform. It extends the separate [window-sum](nsbh-amplitude-window-sum.md)
and [phase](nsbh-explicit-phase.md) checks to the final complex assembly.

## Inputs, conventions, and options

Frequency and three joining centers are positive Hz; total mass is seconds.
The width is positive in $Mf$ units. The three nonnegative dimensionless
carrier amplitudes $P,B,R$ are supplied at each frequency. The positive
strain scale $N$ is supplied; its mass/distance formula is outside this gate.
The phase $\psi$ is in radians and uses an explicit `coalescence` or
`reference_frequency` convention. Inclination $\iota\in[0,\pi]$ is in radians.
The native NSBH plus sign convention is fixed here; no phase or time alignment
is applied to the complex comparisons.

## Mathematics and components

[Thompson et al. (2020), Eqs. (4), (30), Sec. II.2](https://arxiv.org/html/2002.08383v2)
give the three-window amplitude and PhenomD plus NRTidalv2 phase. With
$W_j^\pm=[1\pm\tanh(4M(f-f_j)/d)]/2$,

\[
S=P W_0^-+B W_1^-+R W_2^+ ,\qquad H=N S e^{-i\psi}.
\]

The aligned $(2,\pm2)$ angular factors follow the spin-weighted-harmonic
projection in [Pratten et al. (2020), Eqs. (11)–(13), (53)](https://arxiv.org/html/2004.06503).
In the established NSBH carrier convention the positive-frequency pair is

\[
h_+=-\frac{1+\cos^2\iota}{2}H,\qquad
h_\times=i\cos\iota\,H.
\]

For nonzero $h_+$, their ratio is
$h_\times/h_+=-2i\cos\iota/(1+\cos^2\iota)$, independently of the supplied
amplitude and phase. The phase scalar follows the already versioned
`nsbh.phase.explicit-composition.v1` reference; the remnant/QNM pair and
PhenomD coefficient rows remain conditioning inputs.

## Implementation mapping

`diffgw_reference.nsbh_polarization_composition.compose` evaluates the window
sum and complex projection with mpmath and imports no native waveform code.
The test computes $\psi$ with `diffgw_reference.nsbh_phase.evaluate` from a
conditioned PhenomD specification. Torch and JAX call their existing
`nsbh_amplitude_window_sum` and the new production-called
`nsbh_polarizations_from_amplitude`; their public NSBH paths use the latter
for both output polarizations. A separate public-path ratio check exercises
that connection without assuming the public amplitude is paper-qualified.

## Evidence and limitations

The frozen cases are $(m_1,m_2,\chi_1,\chi_2,\Lambda_2)=(7,1.4,0,0,600)$
and $(12.6,1.4,0.4,0,0)$, $Mf=(.010,.018,.035,.070)$,
$(Mf_0,Mf_1,Mf_2)=(.025,.035,.045)$, $d=.04$,
$(P,B,R)=(.8,.3,.5)$, $N=.73$, and
$\iota=(0,.7,\pi/2,\pi)$. Both explicit phase gauges use
$t_c=.003$ s, $\phi_c=.23$ rad and $Mf_{\rm ref}=.020$ when applicable.
These synthetic carriers and centers probe assembly and angular signs;
they are not physical NSBH fits. CPU FP64 Torch and JAX complex absolute
errors must each be at most $2\times10^{-13}$ against 70-digit scalar
values. The public ratio is checked at 50, 100, 250, 500 and 900 Hz to
$10^{-12}$ absolute. No fitting or output rescaling is permitted. The
[execution receipt](../validation/nsbh_polarization_composition_20260929.md)
records measured errors and adjacent regressions.

The [variant provenance audit](nsbh-amplitude-variant-provenance.md) leaves
three published/native amplitude discrepancies unresolved. The
[remnant guard audit](nsbh-remnant-guard-source-boundary.md) also leaves its
pointwise fallback predicate unsupported by a published equation. This
gate cannot certify upstream carrier formulas, joining decisions, scale,
default legacy phase gauge, remnant extrapolation, or physical waveform
agreement. The independent explicit phase gate checks native phase output;
this gate feeds its scalar phase into the conditioned projection check.

## Reproduce the checks

With mpmath, SymPy, Torch and JAX on CPU FP64, run
`JAX_ENABLE_X64=1 PYTHONPATH=src:tests python -m pytest -q
tests/test_nsbh_polarization_composition_math_reference.py`.
Changing the carrier sign, angular convention, input conditioning or limits
requires a new spec version and evidence receipt.
