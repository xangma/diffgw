# IMRPhenomHM leading (3,3) amplitude ratio

## Scope and status

The component hm.amp.beta33.leading.eq4.v1 covers the leading restricted
stationary-phase (3,3) rescaling in [London et al., arXiv:1708.00404v4,
Eqs. (3)–(4)](https://arxiv.org/html/1708.00404v4), using the independently
printed SPA terms in [Mishra et al., arXiv:1601.05588v1, Eqs. (11), (12a),
(12c)](https://arxiv.org/html/1601.05588v1). It does not claim the complete
HM amplitude, the other modes, or the PhenomD carrier.

## Inputs, conventions, and options

Let $x=M f_{\rm physical}>0$ be the (3,3) Fourier frequency, $g(x)>0$ the
supplied (2,2) amplitude-map frequency in the same geometric units, and
$\delta=(m_1-m_2)/(m_1+m_2)>0$. The mass ordering excludes the equal-mass
zero and isolates the paper's $|\beta|$ without relying on an odd-mode sign
convention. Spins may vary: London retains only the leading (3,3) term, so
this component contains no explicit spin factor. QNM frequencies, the
PhenomD amplitude $A_{22}$, distance, and phase are supplied by the public
assembly, not fitted in this reference. The public hm_mapping_transitions
choice changes $g$ through its amplitude cutoff; both named choices are
tested. The historical default is unchanged.

The map's joins are $f_0^A$ and $f_{33}^{\rm RD}$. The reference checks
one-sided frequency slopes, since the map is continuous but generally not
differentiable at either join. At $x\le0$, or $\delta=0$, the public
quotient used by this gate is outside its domain.

## Mathematics and components

Mishra's $V_m(y)=(2\pi y/m)^{1/3}$ and Eq. (11) give a common SPA
prefactor times $V_m^{-7/2}\hat H_{\ell m}$. At London's restricted order,

$$
\hat H_{22}=-1,\qquad
\hat H_{33}=-\frac34\sqrt{\frac57}\,\delta V_3.
$$

Substitute the full SPA amplitudes, including $V_m^{-7/2}$, into London's
Eq. (4). The second ratio cancels the harmonic-frequency factor, leaving

$$
\beta_{33}(x;g,\delta)=\frac34\sqrt{\frac57}\,\delta
              (\pi g(x))^{1/3},\qquad
\frac{d\beta_{33}}{dx}=\frac{\beta_{33}}{3g(x)}g'(x).
$$

For $x\le f_0^A$, $g=2x/3$; between $f_0^A$ and ringdown,
$g'=(f_{22}^{\rm RD}-2f_0^A/3)/(f_{33}^{\rm RD}-f_0^A)$; above ringdown,
$g'=1$. These are London's Eq. (5) amplitude branches. Eq. (3) uses
$A_{33}(x)\simeq|\beta_{33}(x)|A_{22}(g(x))$. The native odd-mode phase
adapter is not part of this positive-$\delta$ amplitude claim.

| Component | Source | Status | Scalar | Torch/JAX | Gate |
| --- | --- | --- | --- | --- | --- |
| hm.amp.beta33.leading.eq4.v1 | London Eqs. (3)–(4); Mishra Eqs. (11), (12a), (12c) | Executable | diffgw_reference.imrphenomhm_amplitude_ratio | Public D-carrier IMRPhenomHM (3,3) mode | test_imrphenomhm_amplitude_ratio_math_reference.py |

## Implementation mapping

The scalar reference reduces the published leading SPA expressions with
SymPy and evaluates the reduced formula with mpmath. It imports neither
frontend nor a fit table. Torch and JAX production call their own
_production_hhat inside the Eq. (4) path, then multiply the carrier
amplitude at the mapped frequency. The maintained test calls the public
IMRPhenomHM (3,3) mode and separately calls the public PhenomD (2,2)
carrier at $g(x)$. Dividing the unprojected mode magnitude by the carrier
magnitude removes the otherwise opaque carrier, testing the production-called
ratio rather than an unused helper.

The [CPU FP64 validation receipt](../validation/imrphenomhm_amplitude_ratio_20260927.md)
records the integrated public-path run on `len`.

## Evidence and limitations

Frozen CPU-FP64 gate: $m_1=35M_\odot$, $m_2=20M_\odot$,
$\chi_1=0.2$, $\chi_2=-0.1$; both named outer-map choices; samples below,
at, between, and above the two amplitude-map joins. Require relative
ratio-value error $\le 5\times10^{-11}$ with absolute floor
$5\times10^{-13}$. One-sided secant slopes use a geometric-frequency
step $h=10^{-7}$, with relative error $\le 2\times10^{-5}$ and
absolute floor $2\times10^{-8}$. An exact symbolic reduction and a
hand-reduced scalar example check the independent reference. These bounds
test algebra and sampled production composition. They do not validate the
QNM fit, 22 carrier, phase, projections, spins in other modes, calibration,
or full-waveform accuracy. No LAL code or binary output is used.

## Reproduce the checks

Run python -m pytest tests/test_imrphenomhm_amplitude_ratio_math_reference.py
--reference-device=cpu -ra with the repository's reference, torch, jax,
and test dependencies. The strict reference suite discovers this file by
name. A changed source equation, truncation, map convention, or tolerance
requires a new component version and reviewed evidence.
