# Waveform unit constants

Version: **`si-iau-waveform-units/v1`**.

This audit independently checks the three constants used to convert the
TaylorF2 frontend parameters to the mathematical reference's geometric seconds:
`C_SI`, `MSUN_SEC`, and `MPC_SI`. The exact expectations in
[`tests/test_math_reference_units.py`](https://github.com/xangma/diffgw/blob/main/tests/test_math_reference_units.py)
come from the definitions below. Production constants enter only as the values
being checked.

## Definitions and sources

| Component ID | Exact expression | Primary definition |
| --- | --- | --- |
| `units.c_si.v1` | $c=299\,792\,458\;\mathrm{m\,s^{-1}}$ | [BIPM SI metre definition](https://www.bipm.org/en/si-base-units/metre), inspected 22 September 2026 |
| `units.msun_sec.v1` | $T_\odot^{\rm N}=(GM)_\odot^{\rm N}/c^3$, with $(GM)_\odot^{\rm N}=1.3271244\times10^{20}\;\mathrm{m^3\,s^{-2}}$ | [IAU 2015 Resolution B3, arXiv:1510.07674v1](https://arxiv.org/pdf/1510.07674v1), p. 3, solar conversion constants table; exactness specified in recommendation 1, p. 2 |
| `units.mpc_si.v1` | $L_{\rm Mpc}=10^6(648000/\pi)\,149597870700\;\mathrm{m}$ | [IAU 2012 Resolution B2](https://iauarchive.eso.org/static/resolutions/IAU2012_English.pdf), recommendation 1, PDF p. 3 (B2 printed p. 1), defines $1\;\mathrm{au}=149597870700\;\mathrm{m}$ exactly; [IAU 2015 Resolution B2, arXiv:1510.06262v1](https://arxiv.org/pdf/1510.06262v1), p. 4, note 4, specifies $1\;\mathrm{pc}=(648000/\pi)\;\mathrm{au}$ exactly |

The factor $10^6$ converts parsecs to megaparsecs. For component masses expressed
as numerical multiples of the nominal solar mass and distance $d_{\rm Mpc}$ in
megaparsecs, the adapters use

$$
M_{\rm sec}=(m_1+m_2)T_\odot^{\rm N},\qquad
D_{\rm sec}=d_{\rm Mpc}\frac{L_{\rm Mpc}}{c}.
$$

The nominal solar mass parameter is an exact conversion convention. It is not
an exact measurement of the physical Sun. A solar mass in kilograms requires
division by an experimentally determined $G$, with its uncertainty; the reference
therefore constructs $T_\odot^{\rm N}$ directly from nominal $GM$ and exact $c$.
The production expression currently multiplies its `G_SI` and `MSUN_SI` values,
but this test checks the resulting `MSUN_SEC` against the independent nominal
definition. [IAU 2015 B3, recommendations 1 and 5](https://arxiv.org/pdf/1510.07674v1)
distinguish nominal conversion factors from measured properties and SI masses.

## Numerical acceptance and scope

SymPy retains the defining integers, decimal rational, and $\pi$ exactly.
The test evaluates each expectation at 80 and 120 decimal digits and requires
both to select the same binary64 value. It compares the exact rational value
of the production float to the symbolic expectation at both precisions.
`C_SI` must agree exactly. The other two constants may differ from their exact
definitions by at most **half an ulp**, the nearest-rounding limit; here an ulp
is the spacing between binary64 values at the expected magnitude.

On 22 September 2026, all three tests passed. The signed errors
$(\mathrm{production}-\mathrm{definition})/\mathrm{ulp}$ were:

| Constant | Signed error in ulps | Result |
| --- | --- | --- |
| `C_SI` | $0$ | Exact |
| `MSUN_SEC` | Approximately $-0.441393$ | Nearest binary64 value |
| `MPC_SI` | Approximately $-0.167760$ | Nearest binary64 value |

This closes the independent-definition check for these three shared conversion
constants. It does not audit every constant in `diffgw_core`, certify `G_SI` or
`MSUN_SI` individually, remove rounding from subsequent parameter arithmetic,
or establish a physical waveform accuracy bound. The waveform comparisons
continue to use the shared production constants after this separate check.

Run the maintained audit from the repository root:

```bash
python -m pytest tests/test_math_reference_units.py -q
```

SymPy is supplied by `diffgw[reference]`. This optional test module skips when
SymPy is unavailable; the required mathematical-reference gate treats skips as
failures.
