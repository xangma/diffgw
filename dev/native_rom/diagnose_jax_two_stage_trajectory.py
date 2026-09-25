"""Dev-only same-input Torch/JAX physical coarse/fine trajectory comparison."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from dev.native_rom.diagnose_jax_adaptive_physical_event import CASES
from dev.native_rom.eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.integrator import EOBTrajectoryIntegrator as JaxIntegrator
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as JaxFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as JaxHam
from torchwave.approximants.eob.adaptive_driver_clean import (
    AlignedSpinV4Event as TorchEvent,
    integrate_adaptive as torch_integrate,
)
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator as TorchIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as TorchFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as TorchHam

jax.config.update("jax_enable_x64", True)


def run(case_id):
    started = time.monotonic()
    m1, m2, chi1, chi2, omega_start = CASES[case_id]
    delta = (m1 - m2) / (m1 + m2)
    nu = m1 * m2 / (m1 + m2) ** 2

    def tensor(value):
        return torch.tensor(value, dtype=torch.float64)

    th = TorchHam(tensor(nu), tensor(chi1), tensor(chi2), delta=tensor(delta))
    tf = TorchFlux(spin_truncation="tabulated", retain_pan_cubic_spin=False)
    jh = JaxHam(nu, chi1, chi2, delta=delta)
    jf = JaxFlux(spin_truncation="tabulated", retain_pan_cubic_spin=False)
    radius, momentum, angular_momentum = find_quasicircular_initial_conditions(
        th, tf, tensor(omega_start)
    )
    initial_torch = torch.stack((radius, torch.zeros_like(radius), momentum,
                                 angular_momentum)).detach()
    initial_jax = jnp.asarray(initial_torch.numpy())
    torch_field = TorchIntegrator(th, tf).vector_field
    jax_field = jax.jit(JaxIntegrator(jh, jf).vector_field)
    print(case_id, "initial", initial_torch.tolist(), flush=True)

    dt, max_duration, max_step = 0.5, 1000.0, 2.0
    coarse = torch_integrate(
        torch_field, initial_torch, dt, t_end=max_duration,
        event=TorchEvent("coarse"), initial_step=dt, max_step=max_step,
        controller="gsl_state_scaled", atol=1e-10, rtol=1e-9,
    )
    print(case_id, "torch_coarse", float(coarse.time[-1]), flush=True)
    n_coarse = len(coarse.output_time)
    restart_index = (n_coarse - (int(np.ceil(150 / dt)))
                     if (n_coarse - 1) * dt >= 150 else
                     n_coarse - int(np.ceil(n_coarse / 2)))
    refinement = 32
    fine_step = dt / refinement
    fine = torch_integrate(
        torch_field, coarse.output_state[restart_index], fine_step,
        t_end=max_duration, event=TorchEvent("fine"),
        initial_step=fine_step, max_step=max_step,
        controller="gsl_state_scaled", atol=1e-10, rtol=1e-9,
    )
    print(case_id, "torch_fine", float(fine.time[-1]), flush=True)
    jrun = integrate_two_stage_trajectory(
        jax_field, initial_jax, output_step=dt, max_duration=max_duration,
        coarse_initial_step=dt, coarse_max_step=max_step,
        fine_max_step=max_step, atol=1e-10, rtol=1e-9,
    )
    print(case_id, "jax_fine", float(jrun.fine.time[-1]), flush=True)
    torch_time = torch.cat((coarse.output_time[:restart_index],
                            coarse.output_time[restart_index] + fine.output_time[::refinement]))
    torch_state = torch.cat((coarse.output_state[:restart_index],
                             fine.output_state[::refinement]))
    if len(torch_time) != len(jrun.time):
        raise RuntimeError("spliced trajectory output lengths differ")
    state_delta = np.asarray(jrun.state) - torch_state.detach().numpy()
    result = {
        "schema": "diffgw.seobnrv4.jax-two-stage-physical-trajectory.v1",
        "case_id": case_id,
        "status": "diagnostic_complete",
        "scope": "Same initial state and published equations; coarse/fine adaptive trajectory before NQC and attachment",
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "input": {"mass1": m1, "mass2": m2, "chi1": chi1, "chi2": chi2,
                  "omega_start": omega_start, "initial_state": initial_torch.tolist(),
                  "output_step_M": dt, "max_step_M": max_step,
                  "atol": 1e-10, "rtol": 1e-9},
        "torch": {"coarse": {"termination": coarse.statistics.termination[0],
                             "time_M": float(coarse.time[-1]),
                             "accepted": int(coarse.statistics.accepted_steps),
                             "rejected": int(coarse.statistics.rejected_steps)},
                  "fine": {"termination": fine.statistics.termination[0],
                           "time_M": float(fine.time[-1]),
                           "accepted": int(fine.statistics.accepted_steps),
                           "rejected": int(fine.statistics.rejected_steps)},
                  "restart_index": restart_index},
        "jax": {"coarse": {"termination": jrun.coarse.termination,
                           "time_M": float(jrun.coarse.time[-1]),
                           "accepted": jrun.coarse.accepted_steps,
                           "rejected": jrun.coarse.rejected_steps},
                "fine": {"termination": jrun.fine.termination,
                         "time_M": float(jrun.fine.time[-1]),
                         "accepted": jrun.fine.accepted_steps,
                         "rejected": jrun.fine.rejected_steps},
                "restart_index": jrun.restart_index},
        "comparison": {
            "coarse_event_time_abs_M": abs(float(jrun.coarse.time[-1]) - float(coarse.time[-1])),
            "fine_event_time_abs_M": abs(float(jrun.fine.time[-1]) - float(fine.time[-1])),
            "restart_state_max_abs": float(np.max(np.abs(
                np.asarray(jrun.fine.state[0]) - coarse.output_state[restart_index].detach().numpy()))),
            "spliced_samples": len(jrun.time),
            "spliced_state_max_abs": float(np.max(np.abs(state_delta))),
        },
        "elapsed_s": time.monotonic() - started,
        "release_qualified": False,
    }
    result["checks"] = {
        "both_stages_terminate_at_event": all(
            result[backend][stage]["termination"] == "event"
            for backend in ("torch", "jax") for stage in ("coarse", "fine")
        ),
        "restart_index_equal": result["torch"]["restart_index"] == result["jax"]["restart_index"],
        "step_counts_equal": all(
            result["torch"][stage][key] == result["jax"][stage][key]
            for stage in ("coarse", "fine") for key in ("accepted", "rejected")
        ),
        "coarse_event_time_abs_M_at_most": result["comparison"]["coarse_event_time_abs_M"] <= 1e-4,
        "fine_event_time_abs_M_at_most": result["comparison"]["fine_event_time_abs_M"] <= 1e-4,
        "restart_state_max_abs_at_most": result["comparison"]["restart_state_max_abs"] <= 1e-8,
        "spliced_state_max_abs_at_most": result["comparison"]["spliced_state_max_abs"] <= 1e-8,
    }
    result["diagnostic_checks_passed"] = all(result["checks"].values())
    output = Path(f"dev/native_rom/evidence/2026-09-25/jax_two_stage_physical_{case_id}.json")
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(case_id, json.dumps(result["comparison"], sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    selected = parser.parse_args().case
    for name in (CASES if selected == "all" else (selected,)):
        run(name)
