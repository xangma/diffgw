# IMRPhenomT dominant ringdown amplitude

## Scope and status

`T-RD-22-AMP-v1` is a source-level, supplied-input component for the
aligned-spin `(2,2)` ringdown amplitude. Its domain is peak-relative time
`u=(t-t_peak22)/M >= 0`. The only native behavior in scope is the amplitude
returned by the public Torch and JAX `IMRPhenomT.time_domain` carriers and
its local time derivative. This is not a whole-waveform or calibration claim.

## Inputs, conventions, and options

`H_peak=H_22(0)>0` is the dimensionless positive-mode peak amplitude;
`alpha_1>0` and `alpha_2>alpha_1` are QNM damping rates in inverse
total-mass time; and real `d_3` is the fitted shape input. The reference
uses the printed paper branch `d_2=(alpha_2-alpha_1)/2`. The native
carriers accept physical solar masses and seconds, use the same
dimensionless internal `u`, and multiply the mode by their existing
distance/inclination projection. Neither this specification nor its tests
derives the remnant, QNMs, peak fit, or `d_3` fit.

Both carriers also bound `d_2` for some parameters. When that branch is
active, a separate carrier-input check may supply the actual `d_2` to the
same ansatz, but its result is **not** Eq. (27d) fidelity. The test must
identify which branch each sampled binary takes. Native defaults and this
bound are unchanged.

## Mathematics and components

[Estellés et al., *Time domain phenomenological model of gravitational wave
subdominant harmonics for quasi-circular non-precessing binary black hole
coalescences*, arXiv:2012.11923v1, Sec. II C, Eqs. (26), (27c)–(27e)](https://arxiv.org/pdf/2012.11923v1)
gives, at the `(2,2)` peak `u=0`,

$$
\begin{aligned}
d_2&=(\alpha_2-\alpha_1)/2,\\
d_1&=H_{\rm peak}\,\alpha_1\cosh^2(d_3)/d_2,\\
d_4&=H_{\rm peak}-d_1\tanh(d_3),\\
H_{22}^{\rm RD}(u)&=e^{-\alpha_1u}
  [d_1\tanh(d_2u+d_3)+d_4].
\end{aligned}
$$

Direct differentiation gives

$$
H'_{22}(u)=e^{-\alpha_1u}\left[
 d_1d_2\operatorname{sech}^2(d_2u+d_3)
 -\alpha_1\{d_1\tanh(d_2u+d_3)+d_4\}\right].
$$

The algebra requires `H(0)=H_peak` and `H'(0)=0`; these are independent
reference identities. A local stationary point alone does not establish a
global maximum or accuracy against numerical relativity.

| Component ID | Source | Scalar reference | Native carrier | Gate |
| --- | --- | --- | --- | --- |
| `T-RD-22-AMP-v1` | THM Eqs. (26), (27c)–(27e) | `diffgw_reference.phenomt_ringdown_amplitude` | Torch/JAX `IMRPhenomT.time_domain` via `_phenomt_amplitude` | `test_phenomt_ringdown_amplitude_math_reference.py` |

## Implementation mapping

The independent scalar reference uses mpmath and imports no native
waveform, QNM, or fit code. The native state provides the QNM, peak, and
shape inputs. A face-on public waveform exposes the positive mode amplitude
as `hypot(h_plus,h_cross)/(scale*sqrt(5/(4*pi)))`, where
`scale=(M*MSUN_SEC)/(distance*MPC_SI/C_SI)`. The gate differentiates this
public modulus with respect to `u` through `t=u*M*MSUN_SEC` (equivalently,
its seconds derivative times `M*MSUN_SEC`). The arbitrary carrier
phase and reference-frequency phase offset cancel in this modulus.

## Evidence and limitations

The maintained check enforces high-precision peak/derivative identities
on independent scalar inputs, then compares the public Torch and JAX FP64
CPU amplitudes and derivatives at nonnegative ringdown times. Paper-branch
cases separately check `d_2=(alpha_2-alpha_1)/2`; a bounded `d_2`
case is a carrier-algebra diagnostic only. The [execution receipt](../validation/phenomt_ringdown_amplitude_20260927.md)
records the `len` CPU result. The QNM/remnant, peak, and shape
fits, inspiral/merger joins, other modes, physical waveform accuracy, and
whole IMRPhenomT qualification remain outside this evidence.

## Reproduce the checks

On `len` CPU, run
`JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src:tests python -m pytest -q tests/test_phenomt_ringdown_amplitude_math_reference.py --reference-device=cpu -ra`.
The linked execution receipt records the environment, sample coverage,
and results.
