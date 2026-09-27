# NRTidal v1 nonspinning tidal phase

## Scope and status

Specification **`nrtidalv1-nonspinning-tidal-phase/v1`** covers the additive
frequency-domain tidal phase of the dominant `(2,2)`
`IMRPhenomD_NRTidal` carrier for zero component spins, with its merger taper
disabled. The independent scalar reference evaluates both the literal
published Padé relation and the carrier's three-decimal rounding of one
coefficient. This is a component specification, not a whole-waveform or
physical-accuracy certificate.

## Inputs, conventions, and options

The inputs are positive GW frequency $f$ in Hz, component masses $m_A,m_B$ in
solar masses for the native evaluators, nonnegative dimensionless quadrupolar
deformabilities $\Lambda_A,\Lambda_B$, and zero aligned spins. The scalar
reference takes $M_{\rm sec}=G(m_A+m_B)/c^3$ and signed
$\delta=(m_A-m_B)/(m_A+m_B)$. Body labels may be exchanged together with
their deformabilities. $x=(\pi M_{\rm sec}f)^{2/3}$ and
$\eta=(1-\delta^2)/4$. The waveform factor is $\exp(-i\psi_T)$; there is
no tidal phase subtraction at the reference frequency. The native
point-particle baseline cancels in the unit-complex BNS/BBH quotient used
below. Tests use CPU FP64, $f_{\rm ref}=30$ Hz and frequencies at or below
800 Hz, below the carrier's merger taper onset for the declared systems.

`carrier_rounded=False` is the paper-literal scalar default;
`carrier_rounded=True` selects the current native coefficient rounding.
Neither is an alternative native user option. The native call is checked
with `apply_merger_taper=False` to isolate the phase.

## Mathematics and components

[Dietrich, Bernuzzi and Tichy (2017), Eq. (2) and Eq. (7)](https://arxiv.org/pdf/1706.02969v2)
give the effective quadrupolar coupling and frequency-domain Padé phase.
With $X_A=m_A/M$ and $X_B=m_B/M$, substituting
$\Lambda_A=2k^A_2/(3C_A^5)$ into Eq. (2) gives

\[
\kappa^T_{\rm eff}=\frac3{13}\left[
 (X_A+12X_B)X_A^4\Lambda_A+(X_B+12X_A)X_B^4\Lambda_B
 \right]=\frac3{16}\widetilde\Lambda.
\]

The last equality is checked symbolically against the mass-weighted
$\widetilde\Lambda$ of [Wade et al. (2014), Eq. (5)](https://arxiv.org/pdf/1402.5156v1).
The frequency-domain phase is

\[
\psi_T(f)=-\kappa^T_{\rm eff}\frac{39}{16\eta}x^{5/2}
\frac{1+n_1x+n_{3/2}x^{3/2}+n_2x^2+n_{5/2}x^{5/2}}
     {1+d_1x+d_{3/2}x^{3/2}},
\]

where Eq. (7) prints $n_1=-17.428$, $n_{3/2}=31.867$,
$n_2=-26.414$, $n_{5/2}=62.362$, and $d_{3/2}=36.089$.
It constrains $d_1=n_1-3115/1248=-19.923993589\ldots$ so the
low-frequency expansion has the analytic 1PN coefficient $3115/1248$.
The present Torch/JAX carrier and pinned binary instead use $d_1=-19.924$,
the three-decimal rounding of that constraint. This is a documented
source-to-carrier approximation, not exact equality to the analytic PN
constraint. On the declared 30–800 Hz equal/unequal systems it changes the
phase by up to $1.91\times10^{-5}$ rad.

| Component ID/version | Source | Variant | Status | Scalar reference | Torch/JAX | Check |
| --- | --- | --- | --- | --- | --- | --- |
| `nrtidalv1.phase.kappa_eff.v1` | Dietrich Eq. (2); Wade Eq. (5) | paper | executable | `kappa_eff_expr` | `tidal_combinations`, $3\widetilde\Lambda/16$ | symbolic mass/body exchange |
| `nrtidalv1.phase.pade.v1` | Dietrich Eq. (7) | paper literal and rounded carrier | executable | `pade_expr` | `IMRPhenomD_NRTidal` v1 phase branch | PN expansion, coefficient difference |
| `nrtidalv1.phase.tidal.v1` | Dietrich Eq. (7) | additive phase | executable | `phase_expr`, `evaluate` | Torch/JAX v1 phase branch | unit-complex quotient |
| `nrtidalv1.phase.frequency_derivative.v1` | differentiated Eq. (7) | fixed masses/tides | executable | SymPy derivative | Torch/JAX autodiff | 300 Hz derivative |

## Implementation mapping

The [scalar source](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/nrtidalv1_phase.py)
uses SymPy rationals for every printed decimal and mpmath evaluation. It
imports no native or LAL code. The independent Torch and JAX v1 branches in
`imrphenomd_nrtidal.py` multiply their `IMRPhenomD` carriers by the tidal
factor. The [math test](https://github.com/xangma/diffgw/blob/main/tests/test_nrtidalv1_phase_math_reference.py)
compares unit-complex BNS/BBH quotients to the explicit rounded scalar
variant, thereby excluding inherited point-particle amplitude and phase.
The [binary test](https://github.com/xangma/diffgw/blob/main/tests/test_nrtidalv1_phase_lal_binary.py)
uses only LALSuite's public sequence API to form the same unaligned
quotient. No coefficient is inferred from binary outputs.

## Evidence and limitations

The paper math is checked by symbolic $\kappa^T_{\rm eff}$ equivalence,
body exchange, zero tides, and its exact 1PN expansion. Four main cases
$(m_A,m_B;\Lambda_A,\Lambda_B)$ are $(1.35,1.35;400,400)$,
$(1.6,1.2;200,800)$, its body-exchanged copy, and
$(1.35,1.35;0,0)$. Two additional binary holdouts are
$(1.48,1.22;600,250)$ and $(1.25,1.25;900,900)$.
Paper-point frequencies are 30, 100, 300, 500, and 800 Hz; the binary
grid is 30–800 Hz in 5 Hz steps. The predeclared native-to-rounded-scalar
unit-complex residual gate is $10^{-10}$ per point; derivatives at 300 Hz
use relative $10^{-8}$ and absolute $10^{-10}$ rad/Hz. The pinned LALSuite
7.26.15 binary comparison uses relative complex $L_2\le10^{-6}$ and
maximum phase residual $\le10^{-5}$ rad, with no phase/time alignment.

The [27 September 2026 receipt](../validation/nrtidalv1_phase_20260927.md)
records the exact command, versions, cases, and observed results. The
source-literal and carrier-rounded branches remain visibly distinct. These
checks do not qualify the merger frequency or taper, spinning cases,
other modes, the inherited IMRPhenomD carrier, FP32/accelerators, or
physical agreement with neutron-star simulations.

## Reproduce the checks

With CPU FP64 dependencies and LALSuite 7.26.15 installed, run
`DIFFGW_REQUIRE_LAL_BINARY=1 PYTHONPATH=src:tests python -m pytest -q
tests/test_nrtidalv1_phase_math_reference.py
tests/test_nrtidalv1_phase_lal_binary.py`. Strict mode makes absent binary or
reference dependencies fatal. A changed equation or coefficient requires a
new specification version and review of both frontend branches and evidence.
