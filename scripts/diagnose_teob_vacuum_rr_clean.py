#!/usr/bin/env python3
"""Run bounded component, EOM, gradient, and short-step TEOB vacuum checks.

The diagnostic intentionally stops after a few geometric-time units.  It is
an implementation contract check and does not attempt a full inspiral,
NQC extraction, merger attachment, or parity comparison.
"""

import json

import torch

from torchwave.approximants.eob.teob_radiation_clean import TEOBRadiationClean
from torchwave.approximants.eob.teob_vacuum_driver_clean import (
    TEOBVacuumRadiationDriver,
)
from torchwave.approximants.teobresums import (
    TEOBResumSHamiltonian,
    TEOBResumSRadiationReaction,
)


def _number(value):
    value = value.detach().cpu()
    return value.item() if value.numel() == 1 else value.tolist()


def main() -> None:
    dtype = torch.float64
    zero = torch.tensor(0.0, dtype=dtype)
    nu = torch.tensor(0.25, dtype=dtype)
    hamiltonian = TEOBResumSHamiltonian(nu, zero, zero, zero, zero)
    radiation_reaction = TEOBResumSRadiationReaction(
        nu, zero, zero, include_horizon_absorption=True,
    )
    r = torch.tensor(10.0, dtype=dtype)
    pr_star = torch.tensor(-0.003, dtype=dtype)
    pphi = torch.tensor(3.4, dtype=dtype)
    dr_dt, omega, dpr_cons = hamiltonian.equations_of_motion_conservative(
        r, pr_star, pphi,
    )
    h_eff, h_real, h_circ, a_prime = hamiltonian.clean_radiation_inputs(
        r, pr_star, pphi,
    )
    radiation = TEOBRadiationClean(8, residual_phase="zero-unpublished")
    common = dict(
        r=r,
        Omega=omega,
        phi=torch.zeros_like(r),
        prstar=pr_star,
        pphi=pphi,
        Heff=h_eff,
        H=h_real,
        Hcirc=h_circ,
        Aprime=a_prime,
        m1=hamiltonian.m1,
        m2=hamiltonian.m2,
    )
    infinity_flux = radiation.flux(**common, include_horizon=False)
    total_flux = radiation.flux(**common, include_horizon=True)
    _, torque = radiation_reaction.radiation_reaction_forces(
        omega, pr_star, pphi, r=r, hamiltonian=hamiltonian,
    )

    r_grad = r.clone().requires_grad_(True)
    omega_grad = hamiltonian.equations_of_motion_conservative(
        r_grad, pr_star, pphi,
    )[1]
    torque_grad = radiation_reaction.radiation_reaction_forces(
        omega_grad, pr_star, pphi, r=r_grad, hamiltonian=hamiltonian,
    )[1]
    dtorque_dr = torch.autograd.grad(torque_grad, r_grad)[0]

    driver = TEOBVacuumRadiationDriver(
        hamiltonian,
        radiation_reaction,
        include_horizon_absorption=True,
    )
    initial = driver.initial_state(torch.tensor(0.025, dtype=dtype))
    coarse_time, coarse = driver.integrate(initial, 0.5, 8)
    fine_time, fine = driver.integrate(initial, 0.25, 16)
    trajectory = driver.trajectory(torch.tensor(0.025, dtype=dtype), 0.5, 8)

    result = {
        "route": "zero-spin-zero-tide-clean-vacuum",
        "lmax": 8,
        "mode_count": len(radiation.modes(**common)),
        "include_horizon_absorption": driver.include_horizon_absorption,
        "state": {
            "r": _number(r),
            "pr_star": _number(pr_star),
            "pphi": _number(pphi),
            "omega": _number(omega),
        },
        "radiation_inputs": {
            "H_eff": _number(h_eff),
            "H_real_total_over_M": _number(h_real),
            "H_circ_current_pphi": _number(h_circ),
            "A_prime": _number(a_prime),
        },
        "flux": {
            "infinity_angular": _number(infinity_flux),
            "infinity_plus_horizon_angular": _number(total_flux),
            "horizon_delta": _number(total_flux - infinity_flux),
            "reduced_torque": _number(torque),
        },
        "eom": {
            "dr_dt": _number(dr_dt),
            "dpr_star_dt_conservative": _number(dpr_cons),
            "radial_rr": 0.0,
            "dtorque_dr": _number(dtorque_dr),
        },
        "short_trajectory": {
            "duration_M": _number(coarse_time[-1]),
            "coarse_dt_M": 0.5,
            "fine_dt_M": 0.25,
            "max_endpoint_difference": _number(torch.max(torch.abs(coarse[-1] - fine[-1]))),
            "finite": bool(torch.isfinite(fine).all()),
            "frequency_samples": _number(trajectory.frequency),
        },
        "scope": {
            "full_duration": "not attempted",
            "nqc_merger": "not implemented by this bounded driver",
            "parity_claim": False,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

