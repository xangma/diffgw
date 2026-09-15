# Independent adaptive EOB driver

This is a coordinator-written functional specification from the authorized
reference audit. It contains no reference implementation or calibration data.
Implement it independently in Torch, using Fehlberg's published embedded
fourth/fifth-order method: [NASA TR R-315](https://ntrs.nasa.gov/citations/19690021375).
The report is a US government work according to the NASA catalogue. Do not
read GSL or LAL implementation source.

Separate accepted adaptive states from the requested uniform output grid.
Use the Fehlberg fifth-order advance and the embedded fourth-order error
estimate. The standard reference path starts its trial step at the requested
output interval and uses absolute tolerance 1e-10 and relative tolerance
1e-9. A component's error scale is absolute tolerance plus relative
tolerance times its state magnitude. Report the controller and demonstrate
tolerance refinement; a step-size policy is not a waveform calibration.
Retain accepted states and resample each state component with a natural
cubic spline onto times 0, dt, ..., floor(t_last/dt)*dt. The output interval
is not a maximum adaptive step. Keep gradients through accepted arithmetic;
event decisions and sample counts are discrete. Explicitly bound rejected
steps, evaluations and total duration. Do not silently freeze a failed row.

For the non-optimized aligned-spin v4 contract, test termination at accepted
adaptive states, using the already evaluated vector field. The coarse stage
stops when r<6 and either angular frequency decreased from its previous
accepted-state value or the radial-momentum derivative is nonnegative. The
fine stage counts frequency decreases inside r<6 and stops at the fifth
decrease, or at a nonnegative radial-momentum derivative. Invalid derivative
states terminate as an explicitly reported domain event; do not continue
through NaNs or silently treat a failed trajectory as successful. Preserve
the final accepted state in the interpolation input. The event is not the
later spline-refined peak used for attachment.

Run the coarse stage on the requested output grid. If that trajectory lasts
at least 150M, restart from its existing sample at index
N_coarse-ceil(150/dt_output). For a shorter trajectory, restart at index
N_coarse-ceil(N_coarse/2). Validate the index. Keep that sample's state and
orbital phase, and start the fine local time at zero. Its output interval is
dt_output / 2^max(0,ceil(log2(50*dt_output))), at most 0.02M.

Find the attachment epoch from the full fine-grid orbital-frequency spline,
then use the clean NQC/attachment helper. Never override this peak with the
coarse-grid local-polynomial peak. Attach on the fine grid, apply the finite
ringdown support there, and select every power-of-two refinement-factor-th
sample when splicing back into the coarse output. The splice must preserve
the initial phase and both time origins. Keep pre-restart coarse samples and
use the fine waveform from the restart onward.

The existing clean Hamiltonian, flux, initial conditions, mode and attachment
helpers may be used. Keep the current published fixed-step route available
while this optional numerical-compatibility route is validated. Tests must
cover an analytic ODE, rejected steps, accepted-state event timing, natural
cubic resampling, tolerance refinement, batching and a parameter gradient.
Run a short physical EOB check and measure work before launching any full
low-frequency waveform. Neither a passing component test nor a converged
native trajectory alone establishes waveform parity.

## Documented controller compatibility

The first native driver used a different controller from the target. The
[GSL public ODE manual](https://www.gnu.org/software/gsl/doc/html/ode-initval.html#adaptive-step-size-control)
provides the mathematical controller independently of implementation source.
For the state-scaled choice, use the candidate new state in
`D_i = atol + rtol*abs(y_new_i)` and `r = max_i(abs(error_i)/D_i)`.
The published rule has a safety factor of 0.9, a shrink threshold strictly
above 1.1, and a growth threshold strictly below 0.5. Between those thresholds
the step is accepted and its size stays unchanged. A rejected trial uses
`h_new/h = max(1/5, 0.9*r^(-1/q))`; a growth trial uses
`min(5, max(1, 0.9*r^(-1/(q+1))))`, with factor 5 for zero error.

The order here is the public stepper's reported consistency order. The
coordinator queried `gsl_odeiv_step_order` on the installed public RKF45
binary and obtained **q=5**. Its shrink and growth exponents are therefore
`-1/5` and `-1/6`. This API result matters because the manual's generic
4(5) example gives a different order. Twelve public controller probes,
including both threshold boundaries, independently confirm those equations;
see [public-controller.json](verification/eob-adaptive-root-review/public-controller.json).
No coefficients were fitted, and no GSL implementation source was inspected.

Implement an explicitly named controller option with this contract and use it
in the optional EOB compatibility path. Keep the previous controller available
for its existing callers. Test threshold inclusivity, bounded factors,
candidate-state scaling, rejected-state restoration, and an analytic ODE.
Do not require identical floating-point operation ordering or copy a library
implementation. Batch tests must use distinct per-row physical parameters
and different acceptance/event histories; equal rows cannot expose accidental
parameter broadcasting when active rows are selected.
