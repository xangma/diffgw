# NSBH London 220 ringdown input, v1

## Scope and status

This is a source-first component specification for the Kerr `(2,2,0)`
ringdown frequency and quality factor used by the PhenomNSBH amplitude.
Remnant mass and spin are supplied inputs. This gate does not qualify their
Zappa/Jiménez-Forteza provider, the PhenomD phase QNM interpolation, or the
whole waveform.

## Inputs, conventions, and options

`M_f > 0` is the final gravitational mass in solar masses and
`-1 < chi_f < 1` is aligned dimensionless final spin. `f_RD` is in Hz;
`Q` is dimensionless. The complex QNM frequency uses the paper's positive
imaginary damping convention. Convert solar masses to seconds using
`G M_sun/c^3`, with `G=6.67430e-11 m^3 kg^-1 s^-2`,
`M_sun=1.9884098706980507e30 kg`, and `c=299792458 m/s`.

## Mathematics and components

[London and Fauchon-Jones, arXiv:1810.03550v2, Eqs. (21)–(22)](https://arxiv.org/pdf/1810.03550v2)
give the 220 zero-damping polynomial. The same expression and conversion
are printed in [Thompson et al., arXiv:2002.08383v2, Appendix A,
Eqs. (16)–(19)](https://arxiv.org/html/2002.08383v2):

`kappa = sqrt(log(2-chi_f)/log(3))`,

`w = 1 + sum_{n=1}^5 A_n exp(i theta_n) kappa^n`,

where `(A_n, theta_n)` in order are `(1.5578,2.9031)`,
`(1.9510,5.9210)`, `(2.0997,2.7606)`, `(1.4109,5.9143)`, and
`(0.4106,2.7952)`. Then
`f_RD = Re(w)/(2 pi M_f G M_sun/c^3)` and
`Q = Re(w)/(2 Im(w))`.

| Component ID/version | Source | Scalar reference | Native check |
| --- | --- | --- | --- |
| `nsbh.qnm.london-220.v1` | London Eq. (22); Thompson Eqs. (16)–(19) | `diffgw_reference.nsbh_london_qnm` | `test_nsbh_london_qnm_math_reference.py` |

## Implementation mapping

The independent scalar reference evaluates the printed decimal coefficients
as high-precision numbers, assembles the complex polynomial, and applies
the two conversions. Torch and JAX production checks call their existing
`london_ringdown_properties` directly. The scalar module must not import
Torch, JAX, LAL, or native NSBH helpers. Native defaults remain unchanged.

## Evidence and limitations

Frozen before native comparison: `M_f=(2.8,8.4,30.0)` solar masses and
`chi_f=(-0.9,-0.5,0,0.5,0.9,0.99)`, all 18 cross-products. For every
case, 70- and 90-digit scalar results must agree to relative `1e-50` for
both outputs; CPU FP64 Torch and JAX must each satisfy relative error
`<=3e-12` against the 90-digit scalar for both outputs. Inverse mass
scaling and positivity must also hold at this limit. There is no fitting,
phase/time alignment, or binary-derived coefficient. These limits do not
certify the accuracy of the phenomenological fit against Kerr perturbation
solutions; they certify implementation of the printed fit.

The Zappa 2019 PDF v2 Supplementary Table I prints its 24 tidal-factor
coefficients to only about 2–3 significant figures. Jiménez-Forteza 2017
Tables 1–4 and 7–10 also round BBH remnant fit coefficients, whereas native
`bbh_remnant_uib2016` and `zappa_remnant_properties` use additional digits.
An exact-decimal scalar comparison of the full remnant chain would silently
take unpublished precision from an implementation. That dependency remains
conditional until a source-defined rounding interval and variant treatment
are frozen and checked separately. The supplied remnant inputs in this gate
are therefore not claimed to be paper qualified.

## Reproduce the checks

Run `PYTHONPATH=src JAX_ENABLE_X64=1 pytest -q
tests/test_nsbh_london_qnm_math_reference.py` on CPU FP64 with SymPy,
mpmath, Torch, and JAX. The corresponding validation receipt records exact
host, environment, errors, and neighboring tests. Any change to input grid,
fit variant, or tolerance requires a new specification version before
comparison.
