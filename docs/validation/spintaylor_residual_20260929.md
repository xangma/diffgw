# SpinTaylor whole-waveform residual diagnosis (2026-09-29)

**Status:** The direct complex whole-waveform comparison still fails. This
read-only qualification identifies separate carrier and angle-evolution
differences, but does not establish a unique paper-supported production patch.
No LAL or other implementation source, private DCC note, or source-exposed
SpinTaylor commit was consulted.

## Reproduction

Run [`spintaylor_residual_probe.py`](../../scripts/qualification/spintaylor_residual_probe.py)
and [`spintaylor_angle_binary_probe.py`](../../scripts/qualification/spintaylor_angle_binary_probe.py)
from the repository root with `PYTHONPATH=src` in the len PyCBC environment.
The first script accepts `--isolation`, `--modes`, `--mode-sum`, and
`--alpha-shift`. Host: `len`; checkout:
`/tmp/diffgw-spintaylor-cleanroom-20260929`; Python:
`/home/xangma/venv-pycbc/bin/python`; binary versions: `lal 7.4.1`,
`lalsimulation 5.3.1`, Torch `2.13.0+cu130`. Inputs are `(m1,m2)=(36,29)
Msun`, spins `(0.3,0.2,-0.1)` and `(0.1,-0.2,0.2)`, inclination `0.6`,
distance `100 Mpc`, `f_ref=37.3 Hz`, `df=0.25 Hz`, and `20–300 Hz`. The LAL
XPHM binary uses precession version `320` and final-spin setting `2`. All
comparisons use the same positive-frequency bins and complex plus/cross
polarizations.

## Decisive waveform comparisons

Relative L2 errors below normalize by the LAL waveform norm. The fitted
column maximizes complex correlation over time and constant complex scale;
it is a diagnostic, not the direct fidelity gate.

| Comparison | Plus raw | Cross raw | Plus fitted | Cross fitted |
| --- | ---: | ---: | ---: | ---: |
| XHM aligned, all modes | 0.164208 | 0.144298 | 0.163814 | 0.144242 |
| XHM aligned, (2,2) only | 0.009737 | 0.009718 | 0.003284 | 0.003284 |
| XPHM full precession, all modes | 0.253636 | 0.255445 | 0.059416 | 0.060229 |
| XPHM full precession, (2,2) only | 0.248621 | 0.249230 | 0.043954 | 0.043049 |

The aligned (2,2) carrier is close to its binary counterpart. The aligned
all-mode discrepancy is already substantial without precession. With a
single LAL mode-array activation, native aligned `(2,1)` amplitude differs
from LAL by a relative L2 of `10.696`; its strain norm is `0.144` of native
plus versus `0.0131` of LAL plus. The other higher modes also differ.
Single-mode LAL outputs sum to the all-mode plus within `0.0086` relative L2
over the analysis band; native single-mode outputs sum to numerical precision.
This supports a carrier higher-mode contribution without claiming that
single-mode isolation is an exact internal decomposition of the LAL binary.

For the full XPHM waveform, the raw plus error is `0.0253` at 20–60 Hz,
`0.3056` at 60–150 Hz, and `0.9848` at 150–300 Hz. The corresponding fitted
errors are `0.0175`, `0.0711`, and `0.1458`. Removing the native higher modes
does not remove the ~25% raw precessing residual, so the aligned carrier
discrepancy is not its sole cause. Fitting time and phase lowers the full-band
error to ~6%, but the direct waveform gate remains failed.

## Binary angle output and convention boundary

The public binary function `SimIMRPhenomXPSpinTaylorAngles` returns three
length-1121 vectors. Their values at the reference and across frequency
identify them as `(alpha, cos(beta), gamma)`; the API docstring does not name
the vector fields. LAL returns the same alpha for conventions 0 and 1, near
zero at 37.25 Hz, while its gamma changes by about `2.95149` radians. The
native convention-1 alpha at that bin is `1.56751`; native convention 0 is
`-0.00689`. This is a **representation comparison**, not proof that the
angle API returns precisely the angles used by LAL's waveform twist-up.

After removing each series' constant value at 37.25 Hz, the maximum
native–binary difference through 60 Hz is `0.1641 rad` in alpha and
`0.1791 rad` in gamma. The maximum beta difference is `0.0304 rad`.
For example, at 60 Hz LAL has `alpha=1.57286` and `cos(beta)=0.989286`,
whereas native convention 1 has `alpha=3.31032` and
`cos(beta)=0.984530`. These varying differences occur below the native
transition at approximately `66.98 Hz`; a merger–ringdown continuation
alone cannot explain them. They do not isolate a particular RHS coefficient
from the integration method or frame conversion.

An exploratory change subtracting the `1.574396` rad XP geometric alpha
offset from native convention 1, without changing other angles or frames,
*worsened* the all-mode raw plus/cross errors to `1.9705/1.9662`.
Therefore the standalone API's alpha offset cannot be transplanted into
native twist-up as a correction. [Pratten et al., Appendix F Table IV](https://arxiv.org/pdf/2004.06503)
assigns convention 1 the PhenomPv2 alpha offset, but does not publish its
formula; Appendix C Eq. (C16) gives the distinct XP geometric offset.

## Public-equation limit

[Colleoni et al., §II.B, Eqs. (2.24)–(2.25)](https://arxiv.org/pdf/2412.16721)
specify that SpinTaylor alpha and beta enter merger–ringdown with inverse
frequency continuations, beta including an exponential damping factor and a
limit of 0 or pi. They attach at `0.97 f_PN,max`, with `f_PN,max` determined
by the PN integration stop. The native calculation instead attaches at
`0.98 f_MECO` and holds beta constant. The LAL binary's `cos(beta)` rises
from `0.985562` at 150 Hz to `0.994995` at 300 Hz; native stays at
`0.982350`. This is a clear published-shape discrepancy above transition.

The paper does not provide the additional inspiral collocation frequency,
the exact PN stop/integrator settings, or a complete normalization for the
damping factor. It also describes revised coprecessing amplitude fits
without publishing every calibrated coefficient needed to reconstruct the
binary carrier. The [Pratten twist-up equations, Eqs. (3.5)–(3.7)](https://arxiv.org/pdf/2004.06503)
support the native twist-up signs and polarization rotation as written;
these were not changed. No unique binary-faithful update to the reference
gauge, RHS evolution, transition, or carrier follows from the allowed public
material and these probes. The whole-waveform fidelity gate remains open.
