# NRTidalv2 Pv2 inherited remnant-spin component

## Scope and status

Specification **`nrtidalv2-pv2-remnant-spin/v1`** is a conditional component
gate, frozen before comparing the native backends. It does not qualify the
complete BNS waveform or determine its time and reference-frequency origin.

## Inputs, conventions, and options

The aligned remnant parameter is supplied. The bounded domain is finite
positive masses with \(m_1\ge m_2\), \(0\le\chi_p\le 0.5\), and
\(0<a_{f,\parallel}<1\). This restriction avoids using the published
*magnitude* equation to adjudicate the native signed retrograde branch.
Tidal deformabilities do not enter this inherited map.

## Mathematics and components

Dietrich et al. (2019), Sec. III.4, retain the IMRPhenomPv2 precession
dynamics for IMRPhenomPv2_NRTidalv2. The [Pv2 technical note](https://dcc.ligo.org/public/0122/T1500602/004/PhenomPv2_technicalnotes.pdf),
Eqs. (29)–(30), (46), (51), and the text below Eq. (51), specifies the
effective in-plane spin and the implemented remnant-spin choice. For ordered
component masses \(m_1\ge m_2>0\), define

\[
A_1=2+3m_2/(2m_1),\quad A_2=2+3m_1/(2m_2),\quad
\chi_p={\max(A_1m_1^2|\boldsymbol\chi_{1\perp}|,
 A_2m_2^2|\boldsymbol\chi_{2\perp}|)\over A_1m_1^2}.
\]

For a **supplied** positive aligned-spin IMRPhenomD final Kerr parameter
\(a_{f,\parallel}\), the Pv2 prescription assigns all in-plane spin to the
larger body and sets the note's \(\lambda=M_f/M_i\). Equation (46) therefore
reduces to

\[
a_{f,\mathrm{Pv2}}=\sqrt{a_{f,\parallel}^2+
 [\chi_p(m_1/(m_1+m_2))^2]^2}.
\]

The aligned remnant fit and QNM conversion are separate inputs to this gate;
neither is inferred from this equation.

## Implementation mapping

The independent `diffgw_reference.nrtidalv2_pv2_remnant` scalar evaluator
uses arbitrary-precision arithmetic and only the printed equations. The
production Torch and JAX Pv2 drivers call native remnant-spin helpers.

## Evidence and limitations

Three BNS-scale, unequal/equal-mass, double-spin
cases compare the Torch and JAX helpers to it, with absolute error at most
`2e-15` for both \(\chi_p\) and \(a_{f,\mathrm{Pv2}}\), CPU FP64, without
alignment or fitting. A public-BBH waveform smoke/regression check follows
the helper extraction.

This gate does not settle the published angle-map conflict: the Pv2 technical
note Eq. (32) sets the NNLO longitudinal input to the larger body's aligned
spin, whereas the native drivers use a mass-weighted effective spin mapping
selected by earlier binary compatibility checks. The note's Eq. (26) and
[Pratten et al. (2020), Eq. (27)](https://arxiv.org/html/2004.06503) also
print different 2PN orbital-angular-momentum coefficients; the native
drivers use a mixed public-compatibility coefficient. These must remain
distinct from the present remnant-spin component and cannot yet support a
paper-exact public BNS dispatch. The BNS carrier's remnant-frequency spline
time shift and reference-frequency phase origin also require a separate
contract.

Source: [Dietrich et al. (2019), Sec. III.4](https://arxiv.org/html/1905.06011).

## Reproduce the checks

Run `JAX_ENABLE_X64=1 PYTHONPATH=src:tests python -m pytest -q
tests/test_nrtidalv2_pv2_remnant_spin_math_reference.py` on CPU FP64.
