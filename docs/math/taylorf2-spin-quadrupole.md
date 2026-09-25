# TaylorF2 spin-induced quadrupoles

## Scope and status

Specification: **`taylorf2-spin-induced-quadrupole-3.5pn/v1`**. The independent
[SymPy kernel](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_spin_quadrupole.py)
adds neutron-star spin-induced quadrupole corrections to the unchanged
[nonspinning](taylorf2.md), [aligned black-hole spin](taylorf2-aligned-spin.md),
and [adiabatic tidal](taylorf2-tidal.md) references. Each inherited component
keeps its ID and version. The [generated coefficient table](taylorf2-spin-quadrupole-coefficients.md)
comes from the exact expressions.

## Inputs, conventions, and options

Write \(M=m_1+m_2\), \(X_i=m_i/M\), signed \(\delta=X_1-X_2\),
\(\eta=X_1X_2\), and \(v=(\pi M_{\rm sec}f)^{1/3}\). Inputs
`quad_monopole1/2` are the full dimensionless \(C_{Q1,2}\), defined by
\(M_{2,i}=-C_{Qi}m_i^3\chi_i^2\). The Kerr value is **one**; omission of either
input defaults that body to one. Define \(q_i=(C_{Qi}-1)X_i^2\chi_i^2\).

### Frontend contract and verification

Both frontends use the historical integer `spin_order` selector. The reference
selects physical terms explicitly via `quad_terms`:

| `spin_order` | Added quadrupole terms |
| --- | --- |
| 0–3 | None |
| 4–5 | `Q4` |
| 6–7 (7 is the public default) | `Q4`, `Q6` |
| 8 or greater | `Q4`, `Q6`, `Q7_tail`, `Q7_cubic` |
| Negative | `Q4`, `Q6` |

`phenom_d_calibration=True` retains only `Q4` for selectors at least 4 or
negative. The low-level phase default selector is 5. This mapping describes
existing frontend semantics, not a strict PN truncation. The public Torch/JAX
TaylorF2 waveform paths accept either or both quadrupole parameters. The JAX
waveform generator also passes changing quadrupoles through its cached JIT path.

The independent reference retains the inherited Newtonian amplitude,
\(e^{-i\Psi}\) carrier, and both phase conventions. In the coalescence gauge it
adds \(\Delta\Psi(f)\); in the reference-frequency gauge it adds
\(\Delta\Psi(f)-\Delta\Psi(f_{\rm ref})\). Symbolic derivatives hold the other
inputs fixed, with no EOS relation. Inputs to the scalar reference are exact
decimal strings with mass/distance in seconds and frequencies in Hz; frontend
tests convert solar masses/Mpc separately.

## Mathematics and components

The new phase is the **difference** from the inherited Kerr-spin reference:

\[
\Delta\Psi(f)=\frac{3}{128\eta v^5}
\left[Q_4v^4+Q_6v^6+(Q_{7,\mathrm{tail}}+Q_{7,\mathrm{cubic}})v^7\right],
\]

\[
\begin{aligned}
Q_4&=-50(q_1+q_2),\\
Q_6&=\left(\frac{15635}{21}+120\eta\right)(q_1+q_2)
     +\frac{2215}{12}\delta(q_1-q_2),\\
Q_{7,\mathrm{tail}}&=-400\pi(q_1+q_2),\\
Q_{7,\mathrm{cubic}}&=10\sum_{i\ne j}q_i\left[
  \left(X_i^2+\frac{308}{3}X_i\right)\chi_i+
  \left(X_j^2-\frac{89}{3}X_j\right)\chi_j\right].
\end{aligned}
\]

The normalization contains **one** factor \(1/\eta\) outside the bracket.
The 3PN coefficient also equals
\(\frac5{84}\sum_iq_i(9407+8218X_i-2016X_i^2)\); the test proves this
identity symbolically. Simultaneous body exchange leaves all terms invariant.
The correction and its waveform difference vanish exactly at \(C_{Q1}=C_{Q2}=1\)
or zero spins; its derivative with respect to \(C_{Qi}\) need not vanish there.

### Paper audit and scope

[Nagar et al., arXiv:1812.07923v1, Eqs. (2), (38), (44), (46), (47),
(50)–(52)](https://arxiv.org/pdf/1812.07923v1) supply the 2PN/3PN
quadratic-spin and 3.5PN quadratic-spin tail terms. Here their
\(a_i=X_i\chi_i\). Subtracting their Kerr value gives \(Q_4,Q_6,Q_{7,\mathrm{tail}}\)
above. The Kerr limit of their 3.5PN quadratic-spin term also reproduces the
unchanged inherited baseline.

[Dietrich et al., arXiv:1905.06011v2, Eqs. (26)–(27)](https://arxiv.org/pdf/1905.06011v2)
give the quadrupole-dependent cubic-spin term and the equivalent 3PN form.
[Marsat, arXiv:1411.4118v2, Eqs. (6.20)–(6.21)](https://arxiv.org/pdf/1411.4118v2)
independently confirms the cubic-spin coefficient after its SPA conversion.
These are paper-equation audits, separate from Torch/JAX agreement.

## Implementation mapping

The [traceability map](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/spin_quadrupole_traceability.py)
links every new component ID to its pinned paper equations, canonical SymPy
function, Torch/JAX function and coefficient slot, and maintained checks.

## Evidence and limitations

The spin-induced **octupole** is an independent physical parameter. This
specification fixes it to its Kerr value, \(C_{\mathrm{Oc},i}=1\); the omitted
3.5PN contribution would be
\(-440\sum_i(C_{\mathrm{Oc},i}-1)X_i^3\chi_i^3\) inside the same bracket.
No EOS relation connects \(C_{Qi}\) to \(\Lambda_i\). Spin–tidal couplings,
amplitude corrections, precession, higher harmonics, cutoffs, and merger are
outside this slice. Thus the code is a verified **quadrupole correction**, not
a complete spinning-neutron-star waveform or physical-domain validation.

The [frozen 100/160-digit cases](https://github.com/xangma/diffgw/blob/main/tests/data/taylorf2_spin_quadrupole_reference_v1.json)
are converged snapshots of this reference, not an independent oracle. The gate
checks exact paper identities and label exchange; FP64 coefficients, phases,
polarizations, parameter derivatives including the Kerr point; selector and
omission behavior; batching; compilation; and changing JAX generator inputs.

The [dated LALSuite binary comparison](../validation.md#taylorf2-spin-induced-quadrupoles-against-lalsuite-23-september-2026)
independently confirms the 2PN and 3PN non-Kerr coefficients and both complex
polarizations for three matched cases. LALSuite 7.21's TaylorF2 does not expose
the non-Kerr 3.5PN terms at an accepted spin selector, so the paper-equation
and mathematical-reference checks remain the evidence for that slice.

## Reproduce the checks

From the repository root after installing the reference and both frontends:

```sh
python -m pip install -e '.[reference,torch,jax,dev]'
python -m diffgw_reference.spin_quadrupole_artifacts --check
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest tests/test_taylorf2_spin_quadrupole_math_reference.py tests/test_spin_quadrupole_math_reference_traceability.py --require-math-reference --reference-device=cpu -ra
```

The focused tests check this correction and its traceability. The maintained CI
reference gate also runs the full `tests/test_*math_reference*.py` suite in
required mode, including inherited TaylorF2 specifications. A change to the
phase equation or convention needs a new component/specification version and
reviewed regeneration of the coefficient table and frozen fixtures; changing
only test coverage does not change the mathematical version.
