# Public SEOBNRv4 output-grid control

The [predeclared plan](reference_ultrafine_plan.json) and
[pinned result](reference_ultrafine.json) continue the aligned-spin public
binary self-comparison from [the prior campaign](../2026-09-23/README.md).
The same `lal=7.7.1`, `lalsimulation=6.2.1` binary was called at output steps
`0.03125`, `0.0078125`, and `0.001953125 M`. The second run reproduced every
comparison field exactly. These are public-reference diagnostics, not native
waveform validations.

| Public grid refinement | Raw shared-epoch complex L2 | Independent peak-anchor complex L2 |
| --- | ---: | ---: |
| `0.03125 → 0.0078125 M` | 0.00091904 | 0.00021263 |
| `0.0078125 → 0.001953125 M` | 0.00042932 | 0.00048157 |

Neither anchor convention decreases monotonically over the full set of
fourfold refinements, including the two coarser comparisons in the prior
campaign. The public implementation's
[source](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_spin_aligned_e_o_b_8c_source.html)
passes the requested `deltaT` into the low-rate integration/interpolation,
uses it to select the step-back index, and derives the high-rate step from it.
Thus changing `deltaT` changes more than output interpolation. This source
path provides a plausible mechanism for the observed self-differences, but
the diagnostic has not apportioned them among integration, restart, NQC, and
ringdown operations. A single public output step cannot yet be treated as a
numerically converged reference for a tight native parity gate.

Reproduce with `PYTHONPATH=src:. /Users/xangma/repos/diffgw/.venv/bin/python
-m dev.native_rom.diagnose_reference_ultrafine --json-out
/tmp/seobnrv4-reference-ultrafine.json`.

# Historical (3,3) flux placement sensitivity

The [source-formula plan](mode_33_flux_placement_plan.json) and
[same-state result](mode_33_flux_placement.json) isolate a difference between
the native Pan-style Taylor `f33` spin series and the historical non-HM v4
implementation. The latter places the `v^4` through `v^7` Kerr-spin
coefficients inside `rho33`, then cubes it. The local LAL source revision,
file hash, and [published source](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_spin_e_o_b_factorized_waveform_8c_source.html)
are pinned in the plan. This is a source-formula distinction, not a direct
compiled mode-flux comparison.

At four frozen public states (`M Omega=0.032–0.05`), replacing only this
mode's spin residual changes the native torque discrepancy from `13–49 ppm`
to within about `1.5 ppm`. The [extended plan](mode_33_late_flux_plan.json)
and [two-grid result](mode_33_late_flux.json) sample the same public dynamics
at `M Omega=0.06–0.15`. Through `0.12`, the historical choice changes an
approximately `80–812 ppm` discrepancy to at most `13.3 ppm` on both grids.
At `0.15`, about `242–246 ppm` remains. Public torque here is inferred from
a five-point derivative of sampled `p_phi`; these numbers do not establish
historical flux parity or explain the late residual. The same-state native
orbital-frequency derivative differs from the sampled public value by about
`75–76 ppm` at `0.15`, versus at most about `6 ppm` through `0.12`; the native
radial-momentum derivative differs by about `1732 ppm` there. Those late
vector-field gaps need their own Hamiltonian and sampling analysis before
the residual can be assigned to a flux mode.

The first [same-state result](mode_33_flux_placement.json) froze the baseline
source before the diagnostic option was added. Replaying its script with the
current source reproduces every numerical point exactly; the later-flux and
whole-waveform reports record hashes of the implemented choice.

Torch and JAX now expose the historical `(3,3)` placement as an explicit
development choice, with the previous choice retained as the default. The
[frozen whole-waveform plan](mode_33_whole_waveform_plan.json) and
[result](mode_33_whole_waveform.json) use the same 30+20 solar-mass aligned-spin
row, pinned binary, fixed numerical settings, and exact baseline replay.
The sampled-peak complex relative L2 improves from `0.0133609` to
`0.0107870`. With independent quadratic sub-sample peak fits it instead
changes from `0.00807196` to `0.00827564`: the late-inspiral and post-peak
windows improve, but the longer inspiral window worsens. The reversal shows
that this full-waveform comparison is sensitive to peak anchoring; the
candidate is not promoted to the default or to a release model.

Reproduce the latter diagnostics with `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_mode_33_late_flux --json-out
/tmp/seobnrv4-mode33-late-flux.json` and `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_mode_33_whole_waveform --plan
dev/native_rom/evidence/2026-09-24/mode_33_whole_waveform_plan.json
--json-out /tmp/seobnrv4-mode33-whole-waveform.json`. The evidence records
the source and public binary identities. Both commands are diagnostics only.
