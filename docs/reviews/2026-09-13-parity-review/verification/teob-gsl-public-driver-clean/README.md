# Public GSL clean TEOB driver cross-check

[diagnose_teob_gsl_public_driver_clean.py](../../../../../scripts/diagnose_teob_gsl_public_driver_clean.py)
integrates the same clean zero-spin, zero-tide leading-order-horizon vector
field as the paired TEOB diagnostic through the public GSL 2.8
`gsl_odeiv2` interface.  It allocates `gsl_odeiv2_step_rkf45`,
`gsl_odeiv2_control_y_new(1e-13, 1e-11)`, and
`gsl_odeiv2_evolve_alloc(4)`, then calls `gsl_odeiv2_evolve_apply` through a
documented `ctypes` system callback.  The callback evaluates
`TEOBVacuumRadiationDriver(_hamiltonian(), _LeadingOrderHorizonReaction()).vector_field`
without importing any external waveform implementation.

The input is the frozen
`verification/teob-paired-driver-clean/paired-physical-output-lo-horizon.npz`
cache (SHA256
`bdf872bb49873aa6818c6b28cbc17fd601cfcaa37a3ecb7a28e5ebf1e2140dae`).  The
initial state is the final `(pa_radius, pa_phase, pa_pr_star, pa_p_phi)` row:

```
(14.099999999999996, 132.17030522060074,
 -0.0012978484923028056, 4.215698580677255)
```

at `t=9523.94216423246M`.  The first proposal follows the clean continuation
contract, `h=0.5*(pa_time[-1]-pa_time[-2])=43.29817151595944M`.  Integration
is capped at 1000 accepted steps or 6000M and stops at the first accepted
decrease of the second derivative component, `d phase/dt`.

The corrected run completed in 31.16 s with 587 accepted steps and 3,703 RHS
callbacks, all returning `GSL_SUCCESS`.  The first accepted decrease was at
`t=12316.762611274882M`, with `d phase/dt=0.17536980004531028`; it is
`-0.0012737652396026533M` from the supplied public waveform epoch diagnostic
`12316.763885040122M`.  This time comparison is diagnostic only and uses no
waveform assembly, alignment, fitting, or threshold change.

The complete accepted `time`, four-component `state`, frequency, submitted
step proposal, and GSL-returned next proposal are in
`gsl-trajectory.npz` (SHA256
`b8ed4189690007540561b9c1c7320ec753a12294e0511cb44dbe6a3ff3c69229`), with
the structured run and source-hash record in `diagnostic.json` (SHA256
`ff4e38b9978b56e4bd7e3d50cb8765020a455a5bba291c47baac99247f22f3d7`).
The live log is `gsl-driver.log`.

The API semantics used here follow the public GNU GSL ODE documentation:
<https://www.gnu.org/software/gsl/doc/html/ode-initval.html>.
