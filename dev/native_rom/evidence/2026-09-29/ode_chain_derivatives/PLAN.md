# Continuous NQC derivative diagnostic plan

The frozen `tabulated` generator and gates are recorded in
`../ode_gate_slice/README.md`. This diagnostic changes no generator source.

[Bohé et al. (2017), Eq. (7)](https://arxiv.org/html/1611.03703) defines
`N22=(1+sum(a_j q_j))*exp(i sum(b_k p_k))` and the five matching conditions
at the calibrated match time. Let `x=(r,phi,pr*,p_phi)`, `f(x)=dx/dtau` be the
published Hamiltonian plus radiation-reaction vector field, `h(x)` the
published base mode, `A=|h|`, and `psi=-arg(h)` in a local phase chart.
The seven scalar matching inputs are
`g=(A,A*q1,A*q2,A*q3,p1,p2,psi)`. Their exact time derivatives are
`Dg=grad(g) dot f` and
`D2g=f^T Hess(g) f + grad(g) dot (Jac(f) f)`.
The amplitude solve uses values/first/second derivatives of `g[0:4]`;
the phase solve uses first/second derivatives of `g[4:7]`. The phase RHS
is `(Dpsi-target_frequency,D2psi-target_frequency_slope)`. These are the
ordinary derivatives of Eq. (7), evaluated by automatic differentiation;
no finite sampling window or coefficient refit is part of the proposed rule.

Before any production edit, evaluate this vector on the hash-replayed
`ode_2` fine trajectory at its continuous match time, using the natural
cubic state interpolant only to locate that state. Compare every matching
entry and both solved coefficient vectors with the active sampled-spline
system. Independently integrate the same published vector field from the
initial state to that match time using SciPy DOP853 at `(rtol,atol)` of
`(1e-11,1e-12)` and `(1e-12,1e-13)`. Evaluate matching inputs on its dense
solution at symmetric `h=0.1,0.05,0.025 M` five-point stencils, without
reusing the native trajectory interpolant. Compare the fourth-order first
and second differences with chain rule values at the DOP853 center state.
Also compare the two DOP853 center states and chain rule vectors. The
independent integration is a numerical cross-check of the same equations,
not independent physics validation.

Proceed to a production candidate only if both rows yield finite,
well-conditioned solves; the tight two DOP853 states agree within `1e-8`
absolute per state component; the tight DOP853 chain rule matching entries
agree with the native-state chain rule entries within `1e-5` relative using
`max(abs(entry),1e-12)` as scale; and the tight DOP853 `h=0.025 M` stencil
agrees with its center chain rule entries within `1e-4` relative using the
same scale. These bounds screen for sign, branch, differentiation, and state
interpolation errors. They do not constitute a waveform qualification gate.
If satisfied, use one diagnostic candidate that replaces all NQC matching
derivative rows with the continuous chain rule vector, preserving the frozen
match time, attachment boundary, calibration targets, tolerances, and gate
thresholds. Run the original two-row ODE/grid gate without altering its
metrics. Record failures as failures.
