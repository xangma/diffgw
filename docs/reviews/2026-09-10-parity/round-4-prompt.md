Continue implementation in `/Users/xangma/repos/torchwave`, preserving the user’s working tree and all successful fixes. No PR is requested. The original clean-room constraints and full-catalog objective remain in force: use permitted literature and author-released assets, never inspect or translate LAL implementation source, and use its binary API only for validation, not training or parameter tuning.

Independent review reproduced **215 passing tests** and **6/6 cases with `--strict` explicitly enabled**. The previous DC, PSD, doubled-projection and missing-PyCBC controls now fail appropriately. Wrapper spin aliases, eccentric alias routing and the PhenomD autograd derivative test improved. Keep these changes. However, “all five milestones completed” and XAS/XHM `verified_parity` are unsupported. Complete the actual missing implementation and verification below.

Reproductions are in `round-4-evidence/controls.py`, `controls.json`, `model_checks.py`, and `model_checks.json`. Use their configurations and independent negative controls as regressions. The numerical overlap probes are diagnostic, not certification: they independently optimize phase/time for each polarization. Their magnitude relative L2 errors are unaffected by that alignment freedom.

1. **Repair the new API regressions first.**
   - `get_fd_waveform('IMRPhenomXPHM', mass1=30., mass2=20., sample_frequencies=torch.arange(20.,500.,.25,dtype=torch.float64))` raises `TypeError: must be real number, not object`. Supplying only `chi1` also fails. The XPHM branch in `waveforms.py` overwrites the resolved spins with raw `_UNSET` arguments. Use the already resolved values; exercise omitted, one-sided and explicit spins for every registered approximant/alias that supports them.
   - Direct `TaylorF2_Eccentric()(f=f, mass1=30., mass2=20., spin1z=.3)` rejects its documented alias because omitted `chi1` becomes an explicit zero during conflict checking. Direct XAS accepts conflicting `chi1=0., spin1z=.3`. Apply a consistent omitted-versus-explicit policy to supported direct APIs, with tensor-safe checks. Do not claim direct aliases work where a class does not accept them.

2. **Finish strict harness enforcement and independent detector validation.** The new controls are useful, but these additional controls still pass:
   - Replace a non-DC bin below `flow`, or a bin above `fhigh`, with order-one complex strain.
   - Return `NaN` for a mandatory optimized match.
   - Remove every required case: the suite returns `suite_passed=true`, `num_cases=0`.
   - Set `STRICT_PARITY=1`, remove PyCBC through the import control, and call the pytest entry point: it still passes because that path hardcodes `strict=False`.
   - Replace detector antenna factors with arbitrary common constants; both reference and test use the same wrong geometry and still pass.

   Validate all mandatory metrics are finite and present, enforce declared lower/upper support and exact case identities/counts, check oracle as well as test nonfinites, honor strict mode through pytest/CI, and reject unexpected skips. Explicitly enforce the recorded norm-error contract as well. Pass the chosen support options consistently to adapters; all six current passing cases still report `out_of_band_zero=false`.

   Compare H1/L1/V1 antenna factors and time delays against independent references. The harness currently applies only `Fplus*hp + Fcross*hc`; it does not test the detector time-delay factor. Exercise the required named PSDs and reject unknown PSD names. Add durable negative controls for each failure above, preserving the successful existing controls.

3. **Complete XAS mathematics; transcribed unused fits do not implement the model.** In `imrphenom_xas.py`, `total_phase` still comes from `self.model_d(..., return_phase=True)` and is exactly identical to D. The imported inspiral/intermediate/ringdown phase fits are unused. Intermediate amplitude similarly uses `h00*val1 + h01*val4` instead of consuming the imported intermediate-amplitude fits. Trace every required author fit to its actual equation and use; construct the published phase, amplitude regions and matching conditions, including support/termination. Remove the D phase substitution. Validate intermediate quantities and continuity from permitted sources before drawing causal conclusions.

   With the same four published test configurations, `f_ref=20`, `df=.125`, band `[20,512)` Hz and distance 500 Mpc, independent measurements are:

   | masses; aligned spins | Optimized plus match | Magnitude relative L2 |
   |---|---:|---:|
   | 35,30; .2,-.3 | .997625015 | .079324855 |
   | 50,15; .6,.4 | .999395333 | .035713026 |
   | 60,7.5; -.4,.1 | .998313923 | .042769593 |
   | 75,5; .5,-.2 | .993811984 | .030378080 |

   These are not strict parity. Add real XAS cases to the strict harness with a documented convention adapter, both polarizations and shared physical alignment. Use the agreed CPU FP64 direct complex L2/norm gates of `1e-6` and required mismatch gates of `1e-8`. Keep `match>=.99` smoke tests clearly separate. Replace the new XAS batch test’s `atol=1e-12`: at physical strain scale, zero output passes it. Check gradients against converged finite differences, including phase-sensitive quantities.

4. **Finish D and distinguish HM from XHM.** The D amplitude corrections are real progress: the `(30,30,.8,.8)` magnitude error is now **.004308618**, norm ratio **1.000150376**, optimized match **.999974363** for `[20,1024)` Hz, `df=.25`, distance 410 Mpc. Refresh stale measurements; diagnose the remaining error rather than restating the unproven Table V explanation. Add this failure and a deterministic domain matrix to strict validation and finish the mathematics without relaxing gates.

   `IMRPhenomXHM` is still an empty subclass of HM, and both default to the same XAS carrier, producing exactly equal waveforms. Separate model names in a JSON manifest do not change their implementation. Implement their distinct published constructions and oracle tests. At `(40,20,.3,-.1)`, inclination 1 rad, distance 500 Mpc, `f_ref=20`, `[20,512)` Hz and `df=.125`, XHM’s plus optimized match is **.983573322**, magnitude relative L2 **.135737931**; HM gives **.966662026** against its own oracle. Continue the dependent precessing and remaining catalog models under the original scope after these foundations work.

5. **Make completion claims follow the evidence.** Until gates and required coverage pass, mark XAS/XHM incomplete or failed against the strict target. Scope the TaylorF2 results to the actual six configurations. Record full inputs/options, strict mode, case inventory, conventions, versions, command and run identity with immutable results; bind hashes for imported constants, detector/PSD code and data as well as the top-level models. Preserve accurate coefficient/asset provenance. Correct stale numbers and unsupported catalog-wide, ringdown-precision and device claims. Do not invent an external blocker for mathematics that remains to be implemented.

Deliver implemented changes, meaningful regression/gate controls and reproducible before/after measurements. Report exactly which required gates pass or fail. A test count or a relaxed overlap threshold is not completion of the original parity objective.
