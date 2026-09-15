# Independent TEOB vacuum post-adiabatic inspiral

This coordinator-written contract contains equations and numerical/interface
requirements, not reference implementation text or calibration tables. The
implementer must not inspect LAL or TEOB implementation source. Use the existing
clean Torch Hamiltonian and radiation helpers as dependencies. New code is MIT.

The frozen catalog system is 1.4+1.4 solar masses with zero spins **and zero tidal
deformabilities**. Its immediate driver gap does not require tidal calibration.
The reference's ordinary default includes a post-adiabatic (PA) inspiral; the
current short fixed-step native trajectory omits it. Do not change the frozen
case's spin-order option: an explicitly nonspinning reference route can select
different NQC behavior even when the numerical spins are zero.

Use the independent equations in Nagar and Rettegno,
[Rush the inspiral, arXiv:1805.03891v2, Section II](https://arxiv.org/html/1805.03891v2).
Derive the nonspinning specialization and polynomial numerical formulas yourself.
The following notation states that specialization for review. Let `p=p_r*`,
`j=p_phi`, `U=A/r^2`, `Q=sqrt(A/B)`, and `z3=2*nu*(4-3*nu)`. Define

```
Heff = sqrt(p^2 + A*(1 + j^2/r^2 + z3*p^4/r^2))
E = sqrt(1 + 2*nu*(Heff-1))
C = Q*(1 + 2*z3*A*p^2/r^2)/(E*Heff)
dr/dt = p*C
dphi/dt = A*j/(r^2*E*Heff)
dp/dt = -Q*(A' + j^2*U' + z3*p^4*U')/(2*E*Heff)
dj/dt = Fphi
```

Here `E` is `nu*Hhat_EOB`; avoid confusing the two Hamiltonian normalizations.
`Fphi` is the existing reviewed vacuum torque with all 35 positive-m modes
through ell=8 and horizon absorption, radial force zero, no flux NQC. In
particular the circular Hamiltonian input to the radiation helper sets radial
momentum to zero while retaining the current angular momentum. Reject nonzero
spins or tides in this bounded implementation.

Initialize `p=0`, `j=sqrt(-A'/U')` on a decreasing radial grid. Alternate:

```
odd PA order:  p_new = Fphi_old/(j'_old*C_old)
even PA order: j_new = sqrt(-(A' + 2*p*p'*(1+2*z3*A*p^2/r^2)
                              + z3*p^4*U')/U')
```

The odd formula uses the analytic finite `C` at `p=0`. All quantities other
than the momentum being solved are held at the preceding order. Recompute
Hamiltonian/radiation quantities for the next order. Derive the even formula
from `dp/dt=p'*(dr/dt)`, and verify both balance equations independently in
tests. Negative radicands, nonfinite values, noninward motion, or zero balance
denominators are explicit errors, not silently clamped trajectories.

Target numerical choices from the coordinator's functional audit:

- Eight alternating PA orders by default; support even orders 2, 4, 6, 8 for
  convergence diagnostics.
- `r0=max(14, omega_start^(-2/3))`, where `omega_start=pi*M_seconds*f_start`.
  This starting radius is the stated Kepler mapping, not a full frequency solve.
- Radial spacing `dr=0.1`, nominal lower bound 14, and
  `N=floor((r0-14)/dr)+1`, `r_i=r0-i*dr`. The final point need not equal 14.
  If fewer than ten intervals remain, return an explicit skipped-PA result so
  the caller can choose an ODE initial-condition route.
- Momentum derivatives: derivative of the unique quartic through five
  consecutive grid points, centered where possible and using the first/last
  five points at the boundaries. Derive weights from polynomial exactness.
- Time and phase: cumulative integrals of `1/(dr/dt)` and
  `(dphi/dt)/(dr/dt)` along the descending radial grid. Each interval integrates
  the unique cubic through four neighboring nodes: first/last four at the ends,
  and nodes `i-1,i,i+1,i+2` for an interior interval `[r_i,r_(i+1)]`.
  Derive weights from interpolation, without copying source expressions.
- Start time and orbital phase are caller-supplied offsets (default zero).
  Return the full radial/time/state/frequency profiles, endpoint state/time,
  PA order and grid metadata. The later ODE must inherit the endpoint's time
  and phase and can start its trial step at half the last PA time interval.

Implement this PA component only, with vectorized radial evaluations and Torch
gradients through arithmetic. A scalar intrinsic-system API with explicit batch
limitations is acceptable initially. Preserve existing drivers and dispatch.
Do not implement merger/NQC/FD production routing in this subtask. Do not launch
a full low-frequency waveform. A short `r0=20` to approximately 14 radial-grid
probe is permitted after unit tests. Record its runtime and order/grid
refinement, not a claim of LAL waveform parity.

Tests should cover polynomial derivative and integration exactness (including
descending grids and endpoints), independent angular/radial balance checks,
2/4/8-order refinement at weak-field radii, finite parameter gradients, invalid
inputs, skipped PA and endpoint continuity. A test that merely duplicates the
runtime formula is insufficient. The existing fallback regression
`test_teob_factorized_torque_is_din_flux_over_nu_omega` must select nonzero
tidal deformability to exercise the DIN fallback; zero-spin zero-tide inputs
now correctly select the reviewed vacuum radiation route.
