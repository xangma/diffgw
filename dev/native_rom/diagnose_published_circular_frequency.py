"""Dev-only clean-room circular-frequency check for the published v4 22 mode.

The angular partial below is derived from the zero-radial limit of the
published aligned-spin effective Hamiltonian. It uses the native potential and
spin-map values, but takes no numerical derivative and imports no oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import torch

from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as JaxFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as JaxHamiltonian
from torchwave.approximants.eob.seobnrv4_amplitude22 import seobnrv4_published_amplitude22
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as TorchFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as TorchHamiltonian


PLAN = Path(__file__).parent / "evidence/2026-09-25/published_circular_frequency_plan.json"


def circular_frequency_analytic(ham: TorchHamiltonian, r: torch.Tensor,
                                pphi: torch.Tensor) -> torch.Tensor:
    """Return d(H_real/M)/[nu d(p_phi/mu)] at p_r*=0 and fixed r.

    Write C=r²/Lambda_t, Q=sqrt(1+C*p_phi²), and S=S_*(Q²-1). At p_r*=0,
    the published Hamiltonian's G_SO is a rational function of Q, while
    G_SS's p_phi-dependent part simplifies to K*(1-1/Q)/C. Their derivatives
    and the spin-map derivative are applied explicitly below.
    """
    metric = ham.deformed_kerr_potentials(r)
    dt, dr, dt_prime, lam, lam_prime, _ = metric
    a, sigma_star, _ = ham.spin_combinations(r)
    nu = ham._parameter(ham.nu, r)
    b, jr = torch.sqrt(dt), torch.sqrt(dr)
    lapse = torch.sqrt(dt*r.square()/lam)
    nu_r = 0.5*(dt_prime/dt + 2/r - lam_prime/lam)
    b_r = dt_prime/(2*b) - b/jr
    frame_omega = 2*a*r/lam
    omega_r = 2*a*(lam-r*lam_prime)/lam.square()
    c = r.square()/lam
    q = c*pphi.square()
    root = torch.sqrt(1+q)
    dq_dj = 2*c*pphi
    droot_dj = c*pphi/root

    spin = ham.effective_spin(r, q, torch.zeros_like(q))
    n2 = nu.square()
    dspin_dq = (
        sigma_star*(nu/3 - (23*nu+3*n2)*q/36 + (-103*nu+60*n2)/(36*r))
        + a*(nu/4 - 45*nu*q/72 + (-109*nu+51*n2)/(36*r))
    )
    dspin_dj = dspin_dq*dq_dj

    u = (r*lapse-b) - jr*(b_r-2*b*nu_r)
    v = (r*lapse-b) + (b*nu_r-b_r)*jr
    pref = lapse.square()/(r*b.square())
    g_so = pref*(v/root + (u-v)/(root+1))
    dg_so_dj = pref*(-v/root.square() - (u-v)/(root+1).square())*droot_dj

    k = lapse*omega_r*jr/(2*r*b)
    g_ss = frame_omega + b*omega_r*jr/(2*r*lapse) + k*pphi.square()/(root*(root+1))
    dg_ss_dj = k*pphi/root.pow(3)

    dh_eff_dj = (
        frame_omega + lapse*droot_dj
        + (g_so+pphi*dg_so_dj+dg_ss_dj)*spin
        + (pphi*g_so+g_ss-spin/r.pow(3))*dspin_dj
    )
    h_eff = ham.effective_hamiltonian(r, torch.zeros_like(r), pphi)
    return dh_eff_dj/ham.real_hamiltonian_from_eff(h_eff)


def scalar(value) -> float:
    return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)


def run_row(row: dict, steps: list[float]) -> dict:
    dtype = torch.float64

    def t(x):
        return torch.tensor(x, dtype=dtype)

    ham = TorchHamiltonian(t(row["nu"]), t(row["chi1"]), t(row["chi2"]))
    r = t(row["r"]).requires_grad_()
    pr = t(row["pr_star"])
    j = t(row["pphi"]).requires_grad_()
    state = torch.stack((r, t(row["phi"]), pr, j))

    omega_a = circular_frequency_analytic(ham, r, j)
    omega_t = TorchFlux.from_hamiltonian(state, ham)["omega_zero_radial"]
    grad_a = torch.autograd.grad(omega_a, (r, j), retain_graph=True)

    jham = JaxHamiltonian(jnp.asarray(row["nu"]), jnp.asarray(row["chi1"]),
                          jnp.asarray(row["chi2"]))

    def current_jax(radius, angular_momentum):
        jstate = jnp.stack((radius, jnp.asarray(row["phi"]), jnp.asarray(0.),
                            angular_momentum))
        return JaxFlux.from_hamiltonian(jstate, jham)["omega_zero_radial"]

    jr0, jj0 = jnp.asarray(row["r"]), jnp.asarray(row["pphi"])
    omega_jax = current_jax(jr0, jj0)
    grad_jax = jax.grad(current_jax, argnums=(0, 1))(jr0, jj0)

    actual_omega = ham.equations_of_motion_conservative(r, pr, j)[1]
    heff = ham.effective_hamiltonian(r, pr, j)
    hreal = ham.real_hamiltonian_from_eff(heff)

    def amplitude(omega_zero_radial):
        return seobnrv4_published_amplitude22(
            actual_omega, omega_zero_radial=omega_zero_radial,
            heff_over_mu=heff, hreal_over_M=hreal, x1=ham.m1, x2=ham.m2,
            chi1=ham.chi1, chi2=ham.chi2,
        )

    amp_a = amplitude(omega_a)
    amp_t = amplitude(omega_t)
    damp_a = torch.autograd.grad(amp_a, r, retain_graph=True)[0]
    damp_t = torch.autograd.grad(amp_t, r, retain_graph=True)[0]

    def finite_difference(step):
        zero = torch.zeros_like(r)
        high = ham.effective_hamiltonian(r, zero, j+step)
        low = ham.effective_hamiltonian(r, zero, j-step)
        center = ham.effective_hamiltonian(r, zero, j)
        return ((high-low)/(2*step))/ham.real_hamiltonian_from_eff(center)

    ladder = []
    for step in steps:
        fd = finite_difference(step)
        fd_grad_r = torch.autograd.grad(fd, r, retain_graph=True)[0]
        ladder.append({
            "step": step,
            "omega": scalar(fd),
            "omega_absolute_error": abs(scalar(fd-omega_a)),
            "radius_gradient": scalar(fd_grad_r),
            "radius_gradient_absolute_error": abs(scalar(fd_grad_r-grad_a[0])),
        })

    return {
        "name": row["name"], "state": row,
        "omega_analytic": scalar(omega_a), "omega_torch_current": scalar(omega_t),
        "omega_jax_current": scalar(omega_jax),
        "omega_torch_absolute_error": abs(scalar(omega_t-omega_a)),
        "omega_jax_absolute_error": abs(scalar(omega_jax)-scalar(omega_a)),
        "analytic_gradient_r_pphi": [scalar(x) for x in grad_a],
        "jax_current_gradient_r_pphi": [scalar(x) for x in grad_jax],
        "gradient_absolute_errors": [abs(scalar(x)-scalar(y)) for x, y in zip(grad_a, grad_jax)],
        "amplitude_analytic": scalar(amp_a), "amplitude_torch_current": scalar(amp_t),
        "amplitude_absolute_error": abs(scalar(amp_a-amp_t)),
        "amplitude_radius_gradient_analytic": scalar(damp_a),
        "amplitude_radius_gradient_torch_current": scalar(damp_t),
        "amplitude_radius_gradient_absolute_error": abs(scalar(damp_a-damp_t)),
        "centered_difference_ladder": ladder,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    plan_bytes = PLAN.read_bytes()
    plan = json.loads(plan_bytes)
    rows = [run_row(row, plan["centered_difference_steps"])
            for row in plan["frozen_native_states"]]
    report = {
        "schema": "diffgw.seobnrv4.published-circular-frequency-diagnostic.v1",
        "plan": str(PLAN.relative_to(Path(__file__).parents[2])),
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "release_qualified": False,
        "interpretation": "Analytic angular partial of the published native Hamiltonian at p_r*=0; fixed-state numerical and gradient comparison only.",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows),
                      "max_omega_error": max(max(row["omega_torch_absolute_error"],
                                                 row["omega_jax_absolute_error"]) for row in rows),
                      "max_gradient_error": max(max(row["gradient_absolute_errors"])
                                                for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
