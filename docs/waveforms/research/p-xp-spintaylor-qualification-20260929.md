# P, Pv2, XP v223, and SpinTaylor qualification (2026-09-29)

This ledger applies to the pinned local binary on `len`: Python 3.12,
Torch 2.14.0+cpu, JAX 0.11.2 with x64 enabled, LAL 7.7.1, and
LALSimulation 6.2.1. The comparison grid is 0.25 Hz unless specified.
Its checks establish only the listed implementation and binary claims;
none is numerical-relativity validation.

## Public option comparison

| Native route | Public code and explicit option | Observed claim and remaining gap |
| --- | --- | --- |
| `IMRPhenomP` class and dispatcher | Public LAL `IMRPhenomP`; [PyCBC's FD catalog](https://pycbc.org/pycbc/latest/html/waveform.html) lists the named LAL approximants. | Native Torch/JAX P class parity passes selected precessing cases. Direct LAL complex polarizations **fail** the canonical fixture below. The native class alias `IMRPhenomPv2 = IMRPhenomP` does not make this the clean public Pv2 route. |
| Clean `IMRPhenomPv2` dispatcher | Public LAL `IMRPhenomPv2` through `SimInspiralChooseFDWaveform`; [PyCBC](https://pycbc.org/pycbc/latest/html/waveform.html) exposes its LAL route. | The Torch clean driver passes two explicit direct-binary fixtures and five source-frame cases. This does not certify the P alias or all Pv2 parameters. |
| `IMRPhenomXP`, v223 | Public LAL `IMRPhenomXP` with `prec_version=223`, `final_spin_mod=3`, conventions 0/1/6. [Pratten et al.](https://arxiv.org/html/2004.06503v2) give the MSA and frame equations. | One unequal-mass generic-spin, nonzero-observer-azimuth fixture passes the direct complex gate after the declared Appendix-C frame map. The paper-derived zero-nutation endpoint and the v223 binary chart remain distinct; equal-mass and generic near-aligned continuation remain open in the [v223 ledger](xp-v223-reference-closure.md). |
| Native SpinTaylor angles used by `IMRPhenomXPHM-SpinTaylor` | Public LAL `IMRPhenomXPHM` with `prec_version=320`, `final_spin_mod=2`; [Colleoni et al.](https://arxiv.org/html/2412.16721) describe the SpinTaylor construction. [LAL's precession source](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_x__precession_8c_source.html) distinguishes versions 310/311/320/321 and integrates from the reference state toward the low-frequency bound. | Native Torch/JAX reference anchoring, shared-frequency determinism, and selected gradients pass. Full complex LAL comparison fails. The native ODE right-hand side and post-inspiral continuation are not independently equation-qualified. No XHM/XPHM carrier or higher-mode claim follows from angle checks. |

Public software names alone do not identify the same option tuple. In
particular, the P class alias, clean Pv2 dispatcher, XP v223 MSA route, and
SpinTaylor version 320 use different carriers, angle prescriptions, and
reference conventions. [Pratten et al., Eq. (7)](https://arxiv.org/html/2004.06503v2)
give the minimal-rotation identity; [Colleoni et al., Sec. II.2](https://arxiv.org/html/2412.16721)
add a post-inspiral angle prescription. The native SpinTaylor route presently
holds beta at its transition value in merger and ringdown, so its complete
angle continuation is not the published construction. The official LAL
source describes the public algorithm; the installed binary above is the
numerical oracle for the measurements here.

## Fixed fixtures and measured gates

The strict direct complex-polarization gate is relative L2 <= `1e-6` and
unoptimized flat mismatch <= `1e-8`, without phase or time maximization.

- **Pv2:** `(m1,m2)=(35,25) Msun`, spins `(0.2,0.1,0.2)` and
  `(-0.1,0.15,-0.1)`, `f_ref=20`, inclination `0.7`, phase `0.3`, distance
  `500 Mpc`, 20–512 Hz. The direct LAL relative L2 values are
  `4.4183e-14` plus and `4.4091e-14` cross. The held-out `(42,17) Msun`
  case with spins `(0.3,-0.15,-0.3)` and `(0.12,0.08,0.2)`, `f_ref=35`,
  inclination `1.1`, phase `0.71`, gives `3.0161e-14` and `3.1430e-14`.
  Both flat mismatches are at floating-point zero and both direct gates pass.
  PyCBC is absent on this host, so optimized matches were not computed.
- **P:** Replacing both named approximants in the canonical Pv2 fixture with
  `IMRPhenomP` gives direct L2 `1.90610` plus and `1.89529` cross, with
  magnitude L2 `0.07113` and `0.09083`. Both direct gates fail. Cross-backend
  parity of the native class is a separate implementation check.
- **XP v223:** `(35,25) Msun`, spins `(0.2,0.1,0.3)` and
  `(-0.1,0.05,-0.2)`, `f_ref=20`, `phi0=0.4`, `phic=0.3`, inclination `0.7`,
  distance `150 Mpc`, 20–512 Hz. The explicit frame map rotates source spins
  by `Rz(-phi0)` and multiplies the public polarization by `exp(-2i phi0)`.
  Conventions 0, 1, and 6 pass the gate; maximum plus/cross direct L2 among
  them is `1.145e-13`. This is v223 binary compatibility for one generic
  fixture, not a result for other versions or a generic physical validation.
- **SpinTaylor 320:** `(35,25) Msun`, spins `(0.2,0.1,0.1)` and
  `(0,0,-0.1)`, `prec_version=320`, `final_spin_mod=2`, inclination `0.6`,
  distance `100 Mpc`, 20–300 Hz. At `f_ref=20`, direct L2 is `1.87008`
  plus and `1.86983` cross; at `f_ref=37.3`, it is `1.86530` and `1.86067`.
  Both direct gates fail. Phase-optimized flat overlaps are about 0.989–0.990;
  those do not establish complex-strain compatibility. The existing
  PyCBC-dependent overlap test skips on this host.

The SpinTaylor angle bug reproduced with `(36,29) Msun`, spins
`(0.3,0.2,-0.1)` and `(0.1,-0.2,0.2)`, `f_ref=37.3`, and grids
`[37.3,80]` versus `[20,37.3,80]` Hz. Before the change, the same 80 Hz
alpha differed by `0.200869` rad. The reference state was incorrectly used
as the integration state at the grid minimum, and reference offsets selected
the preceding node. The Torch and JAX paths now integrate backward to a
15 Hz anchor (or `f_ref` if lower), forward to the transition, and interpolate alpha/gamma
at `f_ref`. Independent scalar reference-state geometry gives
`theta_J=0.12616479751` rad; integrated beta at `f_ref` is within `3e-6`
rad. The targeted tests require the reference angle gauges below `1e-12`,
cross-backend angle agreement, finite matching spin gradients, and the same
shared-frequency angles to `1e-10`. They do not independently validate the
RHS coefficients or prove RK convergence.

## Reproduction and closure

With the pinned environment and `PYTHONPATH=src:.`, run
`pytest -q tests/test_spintaylor_reference_anchor.py`,
`pytest -q tests/test_phenompv2_sourceframe_clean.py tests/parity/test_phenomp_parity.py`,
and `pytest -q tests/test_xp_phi0_binary_oracle.py`. The Pv2 and P direct
measurements use `tests.parity_harness.run_oracle_comparison` on its
canonical Pv2 and holdout cases, with `strict=False` because PyCBC is absent;
the P diagnostic changes both route names to `IMRPhenomP`. The XP binary
fixture uses `scripts.qualify_xp_xphm_boundaries.mapped_phi0_oracle_case`.

The remaining promotion gates are an independent SpinTaylor RHS and
integration derivation, a converged published post-inspiral continuation,
same-option full complex LAL waveforms across holdouts, and separate physical
validation. P needs an explicit version-matched frame/carrier investigation
before any LAL equivalence claim. No catalog status changes follow from this
bounded ledger.
