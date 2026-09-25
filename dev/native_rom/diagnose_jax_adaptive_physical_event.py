"""Dev-only same-input Torch/JAX accepted-state plunge diagnostic.

This checks the adaptive solver and event on a physical v4 vector field. It
does not establish historical-waveform agreement or release qualification.
"""

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

from dev.native_rom.eob.jax.adaptive_driver_clean import (
    AlignedSpinV4Event as JaxEvent,
    integrate_adaptive_to_time,
    resample_accepted_trajectory,
)
from jaxwave.approximants.eob.integrator import EOBTrajectoryIntegrator as JaxIntegrator
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as JaxFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as JaxHam
from torchwave.approximants.eob.adaptive_driver_clean import (
    AlignedSpinV4Event as TorchEvent,
    integrate_adaptive as torch_integrate,
    resample_natural_cubic,
)
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator as TorchIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as TorchFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as TorchHam

jax.config.update("jax_enable_x64", True)

CASES = {
    "high_start_mixed_spin": (30.0, 20.0, 0.2, -0.1, 0.05),
    "q4_nonspinning": (40.0, 10.0, 0.0, 0.0, 0.04),
    "strong_aligned": (30.0, 20.0, 0.8, 0.6, 0.04),
    "strong_antialigned": (30.0, 20.0, -0.8, -0.6, 0.025),
}


def run(case_id):
    started = time.monotonic()
    dtype = torch.float64

    def tensor(value):
        return torch.tensor(value, dtype=dtype)

    mass1, mass2, chi1, chi2, omega_start = CASES[case_id]
    delta = (mass1 - mass2) / (mass1 + mass2)
    nu = mass1 * mass2 / (mass1 + mass2)**2
    th = TorchHam(tensor(nu), tensor(chi1), tensor(chi2), delta=tensor(delta))
    tf = TorchFlux(spin_truncation="tabulated", retain_pan_cubic_spin=False)
    jh = JaxHam(nu, chi1, chi2, delta=delta)
    jf = JaxFlux(spin_truncation="tabulated", retain_pan_cubic_spin=False)
    r, pr, pphi = find_quasicircular_initial_conditions(th, tf, tensor(omega_start))
    initial_torch = torch.stack((r, torch.zeros_like(r), pr, pphi)).detach()
    initial_jax = jnp.asarray(initial_torch.numpy())
    print(case_id, "initial", initial_torch.tolist(), flush=True)

    torch_field = TorchIntegrator(th, tf).vector_field
    torch_started = time.monotonic()
    trun = torch_integrate(
        torch_field, initial_torch, 0.5, t_end=1000.0,
        event=TorchEvent("coarse"), atol=1e-10, rtol=1e-9,
        controller="gsl_state_scaled", initial_step=0.5,
        max_step=2.0, max_evaluations=10000,
    )
    print(case_id, "torch_done", float(trun.time[-1]), time.monotonic() - torch_started, flush=True)

    jax_field = jax.jit(JaxIntegrator(jh, jf).vector_field)
    jax_started = time.monotonic()
    jrun = integrate_adaptive_to_time(
        lambda _t, state: jax_field(state), initial_jax, 1000.0,
        event=JaxEvent("coarse"), atol=1e-10, rtol=1e-9,
        initial_step=0.5, max_step=2.0, max_evaluations=10000,
    )
    print(case_id, "jax_done", float(jrun.time[-1]), time.monotonic() - jax_started, flush=True)

    dt = 0.5
    jtime, jstate = resample_accepted_trajectory(jrun, dt)
    ttime, tstate = resample_natural_cubic(trun.time, trun.state, dt)
    count = min(len(jtime), len(ttime))
    delta_state = np.asarray(jstate[:count]) - tstate[:count].detach().numpy()
    result = {
        "schema": "diffgw.seobnrv4.jax-adaptive-physical-event.v1",
        "case_id": case_id,
        "status": "diagnostic_complete",
        "scope": "Same initial state, published Hamiltonian and 35-mode flux, adaptive RKF45 coarse event only",
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "input": {"mass1": mass1, "mass2": mass2, "chi1": chi1, "chi2": chi2,
                  "omega_start": omega_start, "initial_state": initial_torch.tolist(),
                  "atol": 1e-10, "rtol": 1e-9, "initial_step": 0.5,
                  "max_step": 2.0, "output_step": dt},
        "torch": {"termination": trun.statistics.termination[0],
                  "time_M": float(trun.time[-1]),
                  "accepted": int(trun.statistics.accepted_steps),
                  "rejected": int(trun.statistics.rejected_steps),
                  "evaluations": int(trun.statistics.evaluations)},
        "jax": {"termination": jrun.termination, "time_M": float(jrun.time[-1]),
                "accepted": jrun.accepted_steps, "rejected": jrun.rejected_steps,
                "evaluations": jrun.evaluations},
        "comparison": {"common_samples": count,
                       "event_time_abs_M": abs(float(jrun.time[-1] - trun.time[-1].detach().numpy())),
                       "max_abs_by_state_component": np.max(np.abs(delta_state), axis=0).tolist(),
                       "rms_by_state_component": np.sqrt(np.mean(delta_state**2, axis=0)).tolist()},
        "elapsed_s": time.monotonic() - started,
        "release_qualified": False,
    }
    result["checks"] = {
        "both_terminate_at_event": result["torch"]["termination"] == result["jax"]["termination"] == "event",
        "accepted_and_rejected_counts_equal": (
            result["torch"]["accepted"] == result["jax"]["accepted"]
            and result["torch"]["rejected"] == result["jax"]["rejected"]
        ),
        "event_time_abs_M_at_most": result["comparison"]["event_time_abs_M"] <= 1e-4,
        "max_abs_uniform_state_at_most": max(result["comparison"]["max_abs_by_state_component"]) <= 1e-8,
    }
    result["diagnostic_checks_passed"] = all(result["checks"].values())
    output = Path(f"dev/native_rom/evidence/2026-09-25/jax_adaptive_physical_event_{case_id}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(case_id, json.dumps(result["comparison"], sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    selected = parser.parse_args().case
    for name in (CASES if selected == "all" else (selected,)):
        run(name)
