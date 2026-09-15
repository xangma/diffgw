# Independent TEOB PA-to-ODE continuation

This is a coordinator-written functional contract. Implement fresh MIT code
from the existing reviewed clean components and public mathematical references.
Do not inspect LAL, TEOB or GSL implementation source or the coordinator's
external audit cache. No calibration data or reference code are supplied here.

Implement an isolated scalar-system trajectory pipeline for zero spins and
zero tides. Use the [PA component](teob-postadiabatic-contract.md) and the
existing clean vacuum Hamiltonian/radiation vector field. The physical equations
are those of [Nagar and Rettegno, Section II](https://arxiv.org/html/1805.03891v2).
Do not change the production waveform dispatch or any calibration provider.

The contract is:

- A completed PA profile transfers its last state, absolute geometric time
  and orbital phase to the ODE exactly. Its first ODE trial has half the last
  PA time interval. The repeated boundary sample appears only once in the
  combined trajectory. A skipped PA profile requires an explicit caller
  initial state and step for now; do not silently use a guessed circular state.
- Use the reviewed Fehlberg 4/5 pair and named state-scaled controller with
  absolute tolerance 1e-13 and relative tolerance 1e-11. Expose tolerances and
  explicit evaluation, rejection, accepted-step and duration limits.
- Detect the first strict decrease of orbital frequency between adjacent
  accepted states. In this zero-spin sector orbital frequency is dphi/dt.
  The event time is that first decreased sample's time, without interpolation
  to a continuous maximum. Return it as distinct metadata for later epoch
  handling. Radius below 1 before that event is an explicit failure to resolve
  the requested peak; do not return it as a complete waveform.
- After the event, continue for at least 2M using a fixed step equal to the
  smaller of 0.1M and the adaptive controller's proposed NEXT step. The 0.1M
  cap is a numerical scheduling choice, not a waveform calibration. Expose a
  cap override for convergence checks. Preserve the controller's next-step
  value as metadata; it is not necessarily the last accepted interval.
- Fixed steps use the fifth-order candidate and the same state-scaled local
  error test, without changing the step. A failed fixed trial or domain error
  must be reported explicitly, retaining accepted-state diagnostics. End at
  the first accepted fixed-step time at or beyond event_time+2M; do not shorten
  the last interval merely to hit exactly 2M. Never silently freeze a state.
- Return the original PA knots and accepted adaptive/fixed ODE knots, state,
  frequency, valid support, peak-event index/time, phase continuity and work
  statistics. A scalar intrinsic API is acceptable; reject unsupported batches
  and spins/tides explicitly. Preserve Torch gradients through arithmetic;
  discrete step/event choices are documented piecewise decisions.

Reuse the reviewed integrator instead of copying its logic. A small additive
next-step metadata field or reusable fixed-step primitive in
`adaptive_driver_clean.py` is allowed if its existing defaults, batch behavior
and SEOB trajectory route remain unchanged. Test any shared addition against
the existing adaptive suite. Coordinate before touching other files.

This subtask stops at a complete dynamics trajectory. It does not implement
NQC, ringdown, mode resampling, observer projection or an FD waveform. Those
must use the accepted trajectory in a separate reviewed assembly task.

Tests must establish exact PA endpoint transfer, no duplicate seam, start-time
and phase offset covariance, accepted-state event semantics, fixed continuation
support, explicit skip/domain/budget handling, and a short physical refinement
check. Use analytically tractable synthetic fields for controller/event tests,
not a second copy of the implementation. After tests, a bounded nu=0.24,
r0=20 PA-to-peak-plus-2M probe is authorized with at most 2000 accepted steps,
30000 evaluations and 6000M ODE duration. Record startup host/cwd/command/PID,
log and stop instruction; avoid running a full low-frequency BNS waveform in
this subtask. Report uncertainties and timing without a LAL waveform-parity
claim.
