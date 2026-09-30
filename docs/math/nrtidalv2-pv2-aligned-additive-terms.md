# NRTidalv2 Pv2 aligned-frame additive finite-size terms

## Scope and status

Specification **`nrtidalv2-pv2-aligned-additive-terms/v1`** covers the
signed tidal-amplitude contribution with the full printed self-spin phase,
before the precession rotation. This is a conditional component-composition
gate, not an `IMRPhenomPv2_NRTidalv2` waveform qualification. It follows
[Dietrich et al. (2019), Eqs. (17)–(21), (24)–(29), Table II and Sec. III.4](https://arxiv.org/html/1905.06011),
with the [printed merger fit](nrtidalv2-amplitude-taper.md). The paper says
to add finite-size corrections to the underlying aligned-spin model before
rotation. It assigns both Eq. (24) and the 3.5PN spin terms to Pv2, whereas
the [D gate](nrtidalv2-paper-d-composition.md) omits both.

## Inputs, conventions, and options

The three `(m1,m2;chi1,chi2;Lambda1,Lambda2)` systems in solar masses are
`(1.35,1.35;0.20,-0.12;350,350)`,
`(1.6,1.2;0.08,-0.12;250,800)`, and the exchanged second system.
Distance is 100 Mpc; the diagnostic D quotient uses inclination 0.4 rad,
zero coalescence phase/time, and `phasing_convention="coalescence"`.
Frequencies are 100 and 800 Hz plus the scalar-fit fractions
`0.75, 0.99, 1.0, 1.025, 1.05, 1.10, 1.15, 1.25` of merger frequency.
CPU FP64 is required.

## Mathematics and components

For positive deformabilities, aligned spins, and coalescence phasing, the
tested signed complex contribution is

\[
T(f)=A_T(f)W(f;f_{\rm mrg})\exp\{-i[\psi_T(f)+S_2(f)+S_3(f)+S_{3.5}(f)]\}.
\]

Here `A_T` has the absolute, distance-dependent Eq. (24) normalization.
Consequently `T` is an **additive amplitude term**, not a BBH amplitude
ratio or a complete strain.

## Implementation mapping

The independent mpmath/SymPy reference is
`src/diffgw_reference/nrtidalv2_pv2_terms.py`, composed from the separate
phase, self-spin, and amplitude references. The native diagnostic composes
the actual paper-D pre-taper BNS/BBH quotient, the native 3.5PN component,
native Eq. (24) amplitude, and native taper in each frontend. It does not
import the reference in Torch or JAX.

## Evidence and limitations

Both Torch and JAX must satisfy
`|T_native-T_ref|/(|A_T,ref| W_ref) <= 1e-10` at nonzero-window bins,
without time/phase alignment, rescaling or fitting. At the strictly
post-taper bin, both must return exact complex zero. The two polarization
routes must agree at the same normalized limit.

No native `IMRPhenomPv2_NRTidalv2` class currently composes these terms
with the aligned-spin carrier and twists them into plus/cross polarizations.
The paper's Sec. III.4 specifies the order of operations but does not turn
this component test into a polarization or precession validation. A full
Pv2 gate still needs that native assembly and a pinned binary or independent
physical comparison with a matching convention. The Fig. 7 versus Eqs.
(28)–(29) Love–Q conflict and the printed-fit versus pinned-LAL merger
boundary difference remain open in the linked component receipts.
The supplied-mode projection operator has a separate
[positive-22 gate](nrtidalv2-pv2-positive-22-projection.md). A subsequent
[normalization gate](nrtidalv2-pv2-tidal-mode-normalization.md) resolves the
Eq. (24) supplied-mode scale. The subsequent [conditional aligned-carrier
gate](nrtidalv2-pv2-aligned-carrier.md) composes this correction with a
supplied BBH carrier; public Pv2 BNS angle and reference conventions remain
unresolved.
The historical `binary_compatible` default is unchanged.

## Reproduce the checks

Run `JAX_ENABLE_X64=1 PYTHONPATH=src:tests python -m pytest -q
tests/test_nrtidalv2_pv2_aligned_additive_terms_math_reference.py` with Torch
and JAX CPU FP64. The execution receipt records the full len command and
dependency versions.

The [execution receipt](../validation/nrtidalv2_pv2_aligned_additive_terms_20260927.md)
records the outcome under this frozen limit.
