# NSBH remnant paper-precision interval

## Scope and status

`nsbh.remnant.rounding_interval/v1` is a **component** qualification of the
Jiménez-Forteza et al. BBH remnant fit followed by the Zappa et al. NSBH tidal
correction. It checks the final mass and spin returned by the native Torch and
JAX remnant providers on five frozen, calibrated, guard-inactive inputs. It is
not a claim of numerical-relativity accuracy, physical fit uncertainty,
extrapolation behavior, or whole-waveform qualification.

## Inputs, conventions, and options

Use $m_{\rm NS}=1.4$ in nominal solar-mass units, $m_{\rm BH}=q m_{\rm NS}$,
$\chi_{\rm NS}=0$, and the ordered $(q,\chi_{\rm BH},\Lambda)$ cases

| Case | $q$ | $\chi_{\rm BH}$ | $\Lambda$ |
| --- | ---: | ---: | ---: |
| A | 3 | 0 | 2500 |
| B | 3 | 0.5 | 2500 |
| C | 5 | 0.5 | 1000 |
| D | 5 | 0.75 | 100 |
| E | 5 | 0.75 | 2500 |

All have $\nu=q/(1+q)^2$ in $[0.109,0.222]$, $\Lambda$ in
$[100,2500]$, and $a_{\rm BH}$ in $[-0.5,0.75]$, the reported Zappa
calibration ranges. Inputs are exact decimal/rational values for the scalar
reference and FP64 values for native evaluation. The NS spin is zero, as in
the Zappa calibration.

## Mathematics and components

[Jiménez-Forteza et al. (2017), arXiv:1611.00332v2](https://arxiv.org/html/1611.00332v2)
give the BBH final-spin fit in Eqs. (7)–(11), (13), (15)–(19), Tables 1–4,
and the radiated-energy fit in Eqs. (21)–(24), (26)–(28), Tables 7–10.
Use $\hat S=(m_1^2\chi_1+m_2^2\chi_2)/(m_1^2+m_2^2)$,
$S=(m_1^2\chi_1+m_2^2\chi_2)/(m_1+m_2)^2$, and
$\Delta\chi=\chi_1-\chi_2$. Their final spin is $S+L'_{\rm orb}$;
their final mass ratio is $1-E_{\rm rad}$. The constrained coefficients
$f_{i3}=64-64f_{i0}-16f_{i1}-4f_{i2}$ for spin and
$f_{i2}=16-16f_{i0}-4f_{i1}$ for energy are derived algebraically, not
independently rounded. The paper's Eqs. (13) and (26) likewise derive spin
$f_{11}$ and energy $f_{10}$ from other printed coefficients; their printed
constraint multipliers carry the same rounding intervals. Explicit zero
coefficients are exact zeros.

[Zappa et al. (2019), arXiv:1903.11622v2](https://arxiv.org/html/1903.11622v2)
give, in Supplementary Eqs. (4)–(6) and Table 1, the mass/spin factor

$$G_F(\nu,a,\Lambda)=
\frac{1+p_1(\nu,a)\Lambda+p_2(\nu,a)\Lambda^2}
     {[1+p_3(\nu,a)^2\Lambda]^2},\qquad
p_k=(p_{k10}a+p_{k11})\nu+(p_{k20}a+p_{k21})\nu^2.$$

The final mass is $(m_1+m_2)(1-E_{\rm rad})G_M$ and the final spin is
$(S+L'_{\rm orb})G_\chi$. For each *printed* decimal coefficient, including
the scaling constants in Jiménez-Forteza Eqs. (7), (8), and (22), define the
closed interval centered on that decimal with half-width **one half of its
last displayed decimal place**. Thus `3.833` denotes `[3.8325,3.8335]`,
and `-1.83e-3` denotes `[-1.835e-3,-1.825e-3]`. This conventional
round-to-nearest interpretation bounds omitted digits; it is not a reported
statistical error or a reconstruction of unpublished coefficients. Exact
integers, the zero constraints, and $\sqrt2,\sqrt3$ remain exact. The scalar
reference propagates every independent printed interval through the paper
equations by conservative interval arithmetic. Correlations from constraints
are preserved by deriving those coefficients from their printed inputs.

## Implementation mapping

The independent scalar interval reference lives in
`src/diffgw_reference/nsbh_remnant_rounding.py`; it imports neither frontend.
The native providers are `bbh_remnant_uib2016` and
`zappa_remnant_properties` in each of
`src/torchwave/approximants/imrphenom_nsbh.py` and
`src/jaxwave/approximants/imrphenom_nsbh.py`. The comparison first checks
that each native *literal* fit coefficient rounds to the matching printed
table entry, then that native BBH and NSBH outputs lie inside the scalar
intervals. This audits implementation of the published fit at its printed
precision, not the provenance of unpublished native digits.

## Evidence and limitations

The case list and the following acceptance conditions were frozen before
native evaluation. For each case, both scalar Zappa-factor intervals must be
strictly below one and both scalar BBH mass/spin intervals positive; otherwise
the gate fails instead of silently testing a guard branch. The native mass
ratio, spin, and final mass/spin must lie within their corresponding scalar
interval, enlarged by **$2\times10^{-12}$ absolute** on each side solely for
FP64 evaluation. No fit-dependent tolerance or outcome-selected cases are
permitted. A source-only interval width is reported for every case. Interval
arithmetic may overestimate the set of jointly attainable outputs, so passing
is a conservative consistency check, not proof of exact coefficient identity.
Paper coefficient standard errors and NR residuals quantify a different
uncertainty and are not substituted for last-digit rounding bounds.

Zappa's supplementary text does not define a complete algorithm for all
extrapolation guards. Consequently this gate excludes those branches, and
does not certify the native fallback/clipping policy, remnant calibration
against NR, derivatives, FP32, or remnant-driven whole waveforms.

## Reproduce the checks

On len CPU FP64 with the current PyCBC-capable environment, run
`PYTHONPATH=src /home/xangma/.cache/diffgw-xp-pycbc-venv/bin/python -m pytest -q tests/test_nsbh_remnant_rounding_math_reference.py`.
The six-heading documentation check is
`PYTHONPATH=src python3 scripts/generate_waveform_library_table.py --check`.
