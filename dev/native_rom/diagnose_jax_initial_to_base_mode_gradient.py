"""Development-only q4 initial-state to uncorrected-mode JAX gradient check."""

import json
import os
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_full_series_nqc_physical import MODE_OPTIONS
from dev.native_rom.diagnose_jax_spliced_whole_waveform import SOURCE_FLUX_OPTIONS
from dev.native_rom.eob.jax.adaptive_driver_clean import (
    integrate_adaptive_to_time,
    replay_accepted_steps,
)
from dev.native_rom.eob.jax.initial_conditions_clean import (
    traceable_quasicircular_initial_conditions,
)
from dev.native_rom.eob.jax.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian,
)
from jaxwave.approximants.eob.initial_conditions import (
    find_quasicircular_initial_conditions,
)
from jaxwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from jaxwave.approximants.eob.published_waveform import (
    published_mode22_and_frequency,
)
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "dev/native_rom/evidence/2026-09-25/jax_initial_to_base_mode_gradient_plan.json"
OUTPUT = PLAN.with_name("jax_initial_to_base_mode_gradient.json")


def main():
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.jax-initial-to-base-mode-gradient-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != {
                "mass1": 40.0, "mass2": 10.0, "chi1": 0.0, "chi2": 0.0,
                "omega_start": 0.04, "varied_parameter": "chi1",
            }
            or plan["finite_difference_steps"] != [0.0001, 0.00001]):
        raise ValueError("frozen gradient plan changed")
    started = time.monotonic()

    def progress(stage):
        print(f"{stage}: {time.monotonic() - started:.1f}s", flush=True)

    case = plan["case"]
    m1, m2 = case["mass1"], case["mass2"]
    nu = m1 * m2 / (m1 + m2) ** 2
    delta = (m1 - m2) / (m1 + m2)
    flux = SEOBNRv4PublishedFlux(**SOURCE_FLUX_OPTIONS)
    omega = jnp.asarray(case["omega_start"])

    def components(chi1):
        ham = SourceCalibratedHistoricalHamiltonian(
            nu, chi1, case["chi2"], delta=delta
        )
        field = EOBTrajectoryIntegrator(ham, flux).vector_field
        return ham, field

    def initial(chi1):
        ham, _ = components(chi1)
        radius, radial, angular = traceable_quasicircular_initial_conditions(
            ham, flux, omega, max_iter=plan["initial_solver"]["fixed_iterations"],
            tol=plan["initial_solver"]["tolerance"],
        )
        return jnp.stack((radius, jnp.zeros_like(radius), radial, angular))

    ham, field = components(jnp.asarray(case["chi1"]))
    eager = np.asarray(find_quasicircular_initial_conditions(ham, flux, omega))
    progress("eager initial complete")
    reference_initial = np.asarray(jax.jit(initial)(case["chi1"]))
    progress("compiled initial complete")
    settings = plan["mesh"]
    run = integrate_adaptive_to_time(
        lambda _time, state: field(state), jnp.asarray(reference_initial),
        settings["duration_M"], initial_step=settings["initial_step_M"],
        max_step=settings["max_step_M"], atol=settings["atol"],
        rtol=settings["rtol"],
    )
    accepted_time = run.time
    progress("adaptive mesh complete")
    mode_options = MODE_OPTIONS | {"retain_pan_cubic_spin": True}

    def result(chi1):
        local_ham, local_field = components(chi1)
        states = replay_accepted_steps(
            lambda _time, state: local_field(state),
            initial(chi1), accepted_time,
        )
        mode, frequency = published_mode22_and_frequency(
            states[-1], local_ham, flux, **mode_options
        )
        return jnp.stack((jnp.real(mode), jnp.imag(mode), frequency))

    compiled = jax.jit(result)
    base = np.asarray(result(case["chi1"]))
    progress("eager replay and mode complete")
    compiled_base = np.asarray(compiled(case["chi1"]))
    progress("compiled replay and mode complete")
    gradient = np.asarray(jax.jit(jax.jacfwd(result))(case["chi1"]))
    progress("compiled spin derivative complete")
    finite_differences = {}
    for step in plan["finite_difference_steps"]:
        estimate = (np.asarray(compiled(case["chi1"] + step))
                    - np.asarray(compiled(case["chi1"] - step))) / (2 * step)
        finite_differences[str(step)] = {
            "derivative": estimate.tolist(),
            "max_abs_error": float(np.max(np.abs(estimate - gradient))),
        }
    checks = {
        "initial_value_max_abs_at_most": (
            np.max(np.abs(reference_initial[[0, 2, 3]] - eager))
            <= plan["checks"]["initial_value_max_abs_at_most"]
        ),
        "frozen_replay_final_state_max_abs_at_most": (
            np.max(np.abs(np.asarray(replay_accepted_steps(
                lambda _time, state: field(state),
                jnp.asarray(reference_initial), accepted_time,
            )[-1]) - np.asarray(run.state[-1])))
            <= plan["checks"]["frozen_replay_final_state_max_abs_at_most"]
        ),
        "compiled_value_max_abs_at_most": (
            np.max(np.abs(compiled_base - base))
            <= plan["checks"]["compiled_value_max_abs_at_most"]
        ),
        "gradient_finite_difference_max_abs_at_most": (
            np.isfinite(gradient).all()
            and all(value["max_abs_error"]
                    <= plan["checks"]["gradient_finite_difference_max_abs_at_most"]
                    for value in finite_differences.values())
        ),
    }
    evidence = {
        "schema": "diffgw.seobnrv4.jax-initial-to-base-mode-gradient.v1",
        "status": "diagnostic_complete",
        "release_qualified": False,
        "plan_sha256": q.sha256(PLAN),
        "source_sha256": {
            str(Path(__file__).relative_to(ROOT)): q.sha256(__file__),
            "dev/native_rom/eob/jax/initial_conditions_clean.py": q.sha256(
                ROOT / "dev/native_rom/eob/jax/initial_conditions_clean.py"
            ),
        },
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "case": case, "accepted_steps": run.accepted_steps,
        "rejected_steps": run.rejected_steps,
        "termination": run.termination, "initial_eager": eager.tolist(),
        "initial_traceable": reference_initial.tolist(),
        "base_mode_real_imag_frequency": compiled_base.tolist(),
        "spin_derivative": gradient.tolist(),
        "finite_differences": finite_differences,
        "checks": {key: bool(value) for key, value in checks.items()},
        "diagnostic_checks_passed": bool(all(checks.values())),
        "elapsed_s": time.monotonic() - started,
    }
    OUTPUT.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": evidence["status"], "checks": evidence["checks"],
        "gradient": evidence["spin_derivative"],
        "finite_differences": evidence["finite_differences"],
    }, sort_keys=True), flush=True)
    return 0 if evidence["diagnostic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
