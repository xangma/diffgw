# q=10 TP/TPHM precession gate decision (2026-09-27)

Using the frozen measurements at qualification head `5380379`, no bounded,
source-backed production correction can be selected for the two q=10
precession failures. The case is
`(m1,m2)=(60,6)`, spins `(0.2,0.1,0.8)` and `(0.1,0,0.1)`, `f_ref=20 Hz`.
The matched-option [holdout receipt](phenomt_holdouts_torch_20260925.json)
records TP/TPHM MSA relative L2 `0.0202333/0.0225165` against the retained
`0.01` gate and numerical PV300/FS4 L2 `0.308418/0.307103` against the
retained `0.03` gate. MSA mismatch is `0.000200/0.000249` versus `0.0001`;
PV300/FS4 mismatch is `0.04753/0.04700` versus `0.0005`. The strict expected
failures in `tests/test_phenomt_family_lal.py` remain the release gates; none
is widened or removed.

| Path | Earliest demonstrated discrepancy | Source-backed decision | Exact missing dependency |
| --- | --- | --- | --- |
| MSA PV223 | The [scalar-angle receipt](phenomt_msa_scalar_angles_20260925.json) finds `0.0162258` rad relative rotation across 5,326 samples strictly before MECO. Production and independent published-equation scalar angles agree below `3e-10`. A separately audited compatibility path drops the public residual to `3.32e-6` rad, beginning with initial-reference spin contractions instead of precession-averaged radiation-reaction contractions; a later `q²` quadratic-nutation phase convention also changes. | Retain the published equations. [Chatziioannou et al., Appendix A, Eqs. (90)–(95)](https://arxiv.org/html/1703.03967) explicitly prescribe precession-averaged couplings. Binary agreement alone does not authorize the contrary substitution. | An independent author specification, derivation, or licensed author artifact for the **PV223 implementation convention** that explicitly resolves both the flux spin contractions and quadratic-nutation phase convention. Then recheck pre-MECO angles and the frozen whole-strain gate. |
| Numerical PV300/FS4 | The [trajectory receipt](phenomt_independent_trajectory_20260925.json) finds native and independent published RHS/transport agreement below `4.78e-9` rad, but public Euler-derived rotation diverges before merger. The [clock/chart receipt](phenomt_pv300_clock_chart_20260926.json) finds public carrier frequency agreement within `6.53e-7` relative and minimal-rotation gamma residual `1.70e-11` rad over 2 s, while the reference-event orbital-axis speed is `0.516152/s` publicly versus `0.446028/s` from the published RHS. | Retain the published spin and Kerr quadrupole terms; output agreement does not identify a term to change. [Estellés et al., Sec. II.4.1, Eqs. (12)–(14)](https://arxiv.org/html/2105.05872) specifies the carrier and evolution structure but delegates higher-order spin evolution to SpinTaylor infrastructure; [Sturani](https://dcc.ligo.org/public/0122/T1500554/023/dLdS.pdf) provides the independent PN spin equations. | An independently documented **PV300 precession state/time-map and Euler construction**, or author-provided raw `v(t)`, spin vectors and orbital-axis series for this fixed input. Those would separate a hidden precession clock, spin/axis RHS, and alpha/beta chart before changing an equation. |

These dependencies are model-compatibility provenance, not a request to fit
the public binary. The public outputs expose angles and co-precessing modes,
but not the intermediate PV300 state. No LAL implementation source or
constants were used, and no waveform equation, coefficient, or whole-strain
acceptance threshold is changed by this decision receipt. The detailed
component checks and limits remain in the [validation report](phenomt.md).

The maintained q=10 scalar gate in
`tests/test_clean_room_msa.py::test_q10_msa_flux_spin_couplings_obey_published_scalar_averages`
now evaluates the six contractions in Appendix A (90)–(95) with 50-digit
`mpmath` arithmetic, taking the separately checked MSA roots and `c1` as
inputs. The published-equation flux path passes this isolated downstream gate;
the initial-reference PV223 compatibility path differs already in the first
spin projection by more than `5e-4`. This establishes which path follows the
published averaging prescription, but cannot identify an undocumented PV223
implementation convention.

The adjacent
`tests/test_clean_room_msa.py::test_q10_msa_appendix_c_quadratic_phase_diagnostic`
evaluates [Appendix C, Eqs. (116)–(118)](https://arxiv.org/html/1703.03967)
directly with 50-digit scalar arithmetic and the same upstream `c1` and
averaged spin magnitude. For this case the printed Eq. (117) gives
`psi2=9.7078492509`; the limiting-root equation path gives `9.4134154957`
and the PV223 compatibility path gives `7.3367234407`. Eq. (116) agrees with
the equation path at `2e-12`, and the q=10 case passes the separate
small-velocity cubic-root expansion check in
`tests/test_precession_msa_pn_clean.py`. Both evaluations use the same
upstream `c1` and averaged spin magnitude; the two factors under Eq. (118)'s
square root are negative at this input, so the test uses the positive real
root of their positive product. The cause of the remaining Eq. (117) versus
root-expansion difference is unresolved: no particular input, branch,
printed term, or approximation has been established as responsible. The
printed formula therefore does not independently validate either path or
justify replacing a coefficient or relaxing the whole-waveform gates.

A defensible next experiment needs an author specification of the PV223
convention and a reconciliation of Eq. (117) with the cubic-root expansion,
or, for PV300, author-provided intermediate state at the same frozen q=10
input.
