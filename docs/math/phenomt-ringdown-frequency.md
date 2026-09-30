# IMRPhenomT dominant ringdown frequency and phase

## Scope and status

`T-RD-22-FREQ-PHASE-C4ZERO-v1` covers the aligned-spin `(2,2)` ringdown
angular frequency, its phase increment, and frequency slope for supplied
peak frequency, QNM frequency, and shape parameters. It checks the actual
Torch and JAX carrier evaluation on three binary states. The status is
**qualified for the peak-matched sign branch**, with the printed source
inconsistency below retained as an explicit qualification boundary. Neither
the parameter-space calibration nor the full waveform is qualified here.

## Inputs, conventions, and options

Time `u=(t-t_peak22)/M` is dimensionless and `u>=0`; the `(2,2)` amplitude
peak is `u=0`. The supplied inputs are dimensionless angular frequencies
`omega_peak=omega_22(0)` and `omega_RD`, a positive inverse-time shape
parameter `c2`, and positive dimensionless `c3`. The checked states have
`omega_RD>omega_peak`; this is the increasing-from-below branch. The paper
sets `c4=0` and treats `c2` and `c3` as calibration outputs. Native
`final_spin` overrides change `omega_RD` but are outside this gate.

## Mathematics and components

[Estellés et al., IMRPhenomTHM, arXiv:2012.11923v1, Sec. II.3, Eqs. (24)–(27),
and Sec. II.4, Eqs. (28), (31b), (31e)](https://arxiv.org/pdf/2012.11923v1)
defines the ringdown ansatz and peak matching. The **printed equations have
an internal sign inconsistency**: Eq. (25) writes
`omega−omega_RD=+c1*c2*c3*exp(−c2*u)/(1+c3*exp(−c2*u))` when `c4=0`, while
Eq. (27a) sets `c1=(1+c3)*(omega_RD−omega_peak)/(c2*c3)`. Inserting
`u=0` into the printed Eq. (25) gives `omega(0)=2*omega_RD−omega_peak`,
not `omega_peak`. Eqs. (28), (31b) require the ringdown and merger
frequencies to agree at the peak. Thus this specification **infers** the
minus sign in Eq. (25) while preserving Eq. (27a) and the stated join:

$$
c_1=\frac{(1+c_3)(\omega_{\rm RD}-\omega_{\rm peak})}{c_2c_3},\qquad
E(u)=e^{-c_2u},\qquad
\omega(u)=\omega_{\rm RD}-\frac{c_1c_2c_3 E(u)}{1+c_3E(u)}.
$$

Integrating `d phi/du=omega` with `phi(0)=0`, and differentiating the
frequency, gives

$$
\Delta\phi(u)=\omega_{\rm RD}u+c_1\log\!\frac{1+c_3E(u)}{1+c_3},\qquad
\omega'(u)=\frac{c_1c_2^2c_3E(u)}{(1+c_3E(u))^2}.
$$

These formulas yield `omega(0)=omega_peak`, `Delta phi(0)=0`, positive
slope, and `omega(u)→omega_RD` as `u→∞`. The sign choice is an algebraic
resolution of contradictory printed statements, **not** a claim that the
paper literally prints this branch or that an erratum establishes it. One
could instead reverse Eq. (27a)'s `c1` sign and retain Eq. (25)'s plus;
that gives the same observable `omega(u)` but a different coefficient
convention. The source does not establish which printed line carries the
typo, so the check explicitly adopts the native Eq. (27a) convention.

| Component ID | Source | Reference | Native production path | Check |
| --- | --- | --- | --- | --- |
| `T-RD-22-FREQ-PHASE-C4ZERO-v1` | THM Eqs. (25), (27a), (28), (31b,e), with documented sign inference | `diffgw_reference/phenomt_ringdown_frequency.py` | both `IMRPhenomT._coeff` and `_evaluate`; `_domega_full` | `test_phenomt_ringdown_frequency_math_reference.py` |

## Implementation mapping

The scalar mpmath reference imports no native implementation or fit. The
gate extracts supplied `omega_peak`, `c2`, `c3`, and `omega_RD` from each
frontend's production `_coeff` state; checks its peak-derived `c1`; then
compares production-called `_evaluate` frequency and phase increments and
`_domega_full` slope at six ringdown times. It subtracts the native
constant phase at `u=0`, so only the covered phase increment is compared.
The native QNM and calibration providers are input providers in this test,
not independent source evidence for their coefficients.

## Evidence and limitations

The scalar reference separately checks the peak identity, its analytic
phase against a high-precision integral, and slope against numerical
differentiation. The frontend gate uses FP64 CPU and tolerances
`rtol=3e-12`, `atol=1e-13` for frequency, slope and `c1`, and
`atol=2e-12` for phase. The [execution receipt](../validation/phenomt_ringdown_frequency_20260927.md)
records the tested versions and result. This confirms only the supplied
input ringdown algebra on the three declared states. It does not establish
the printed Eq. (25) literally, QNM/remnant or `c2,c3,omega_peak` fit
provenance, native override behavior, amplitude, merger/inspiral joins,
other modes, or physical agreement with NR.

## Reproduce the checks

On `len`, run
`JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src:tests python -m pytest -q tests/test_phenomt_ringdown_frequency_math_reference.py --reference-device=cpu -ra`.
The strict math-reference CI gate should collect the file without skips.
