"""Dev-only, per-row independent Torch/JAX source-backed waveform parity.

Run each predeclared row in a separate process so a failed row retains its
partial evidence and cannot prevent the other row from being evaluated.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import torch

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from dev.native_rom.diagnose_jax_adaptive_physical_event import CASES
from dev.native_rom.diagnose_source_aligned_native_cap_full_crop_v5 import final_h22_crop
from dev.native_rom.diagnose_source_aligned_ode_chain_amplitude_v2 import (
    _amplitude_hook, _continuous_amplitude,
)
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_matrix import _phase_matrix
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_rhs import _rhs_derivatives
from dev.native_rom.eob.jax import numerical_attachment_clean as ja
from dev.native_rom.eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from torchwave.approximants.eob import numerical_attachment as ta
from torchwave.approximants.eob.adaptive_driver_clean import AlignedSpinV4Event, integrate_adaptive
from torchwave.approximants.eob.nqc import seobnrv4_nqc_targets as torch_targets


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PROTOCOL = EVIDENCE / "warm_jax_two_row_domain_protocol_v1.json"
EXECUTION = EVIDENCE / "warm_jax_two_row_domain_execution_v1.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def compare(left, right):
    a, b = array(left), array(right)
    n = min(len(a), len(b))
    if n < 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("empty or nonfinite comparison")
    denominator = np.linalg.norm(b[:n])
    if denominator <= 0:
        raise ValueError("zero reference comparison norm")
    return {"left_samples": len(a), "right_samples": len(b), "common_samples": n,
            "relative_l2": float(np.linalg.norm(a[:n] - b[:n]) / denominator),
            "max_abs": float(np.max(np.abs(a[:n] - b[:n])))}


def control_sha(controls):
    digest = hashlib.sha256()
    for name in sorted(controls):
        values = np.ascontiguousarray(array(controls[name]), dtype="<f8")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def rk4(field, state, step):
    k1 = field(state)
    k2 = field(state + (step / 2) * k1)
    k3 = field(state + (step / 2) * k2)
    k4 = field(state + step * k3)
    return state + (step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def checked_inputs(case_id):
    protocol = json.loads(PROTOCOL.read_text())
    execution = json.loads(EXECUTION.read_text())
    if (protocol["schema"] != "diffgw.seobnrv4.warm-jax-two-row-domain-protocol.v1"
            or protocol["release_qualified"] is not False
            or execution["schema"] != "diffgw.seobnrv4.warm-jax-two-row-domain-execution.v1"
            or execution["release_qualified"] is not False
            or execution["protocol_sha256"] != sha(PROTOCOL)
            or execution["script_sha256"] != sha(__file__)):
        raise ValueError("frozen protocol or runner changed")
    for name, item in protocol["inputs"].items():
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen input changed: {name}")
    cases = {row["case_id"]: row for row in protocol["rows"]}
    if (case_id not in cases or list(CASES[case_id]) != cases[case_id]["parameters"]
            or list(cases) != execution["case_order"]):
        raise ValueError("case changed from frozen protocol")
    source = protocol["source_options"]
    if (source["hamiltonian_calibration"] != "lal_v4_rounded"
            or source["hamiltonian_log_derivative"] != "lal_v4"
            or {key: value for key, value in source.items() if key not in
                ("hamiltonian_calibration", "hamiltonian_log_derivative")}
            != driver.SOURCE_FLUX_OPTIONS):
        raise ValueError("source-backed formula choice changed")
    return protocol, execution, cases[case_id]


def run(case_id):
    protocol, execution, row = checked_inputs(case_id)
    output = EVIDENCE / execution["outputs"][case_id]
    if output.exists():
        raise FileExistsError(f"evidence already exists: {output}")
    torch.set_num_threads(1)
    started = time.monotonic()
    result = {"schema": "diffgw.seobnrv4.warm-jax-two-row-domain.v1",
              "status": "running", "release_qualified": False, "case_id": case_id,
              "row_tuple": row["parameters"], "settings": protocol["settings"],
              "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
              "protocol_sha256": sha(PROTOCOL), "execution_sha256": sha(EXECUTION),
              "script_sha256": sha(__file__), "stages": {}, "checks": {}}

    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output)

    save()
    try:
        m1, m2, chi1, chi2, requested = row["parameters"]
        nu, delta = m1*m2/(m1+m2)**2, (m1-m2)/(m1+m2)

        def tensor(x):
            return torch.tensor(x, dtype=torch.float64)

        th = driver.TorchSourceHam(tensor(nu), tensor(chi1), tensor(chi2), delta=tensor(delta))
        tf = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
        jh = driver.JaxSourceHam(nu, chi1, chi2, delta=delta)
        jf = driver.JaxFlux(**driver.SOURCE_FLUX_OPTIONS)
        tr, tp, tl = driver.find_quasicircular_initial_conditions(th, tf, tensor(requested))
        jr, jp, jl = driver.jax_initial_conditions(jh, jf, jnp.asarray(requested))
        ts = torch.stack((tr, torch.zeros_like(tr), tp, tl)).detach()
        js = jnp.stack((jr, jnp.zeros_like(jr), jp, jl))
        tfield = driver.TorchIntegrator(th, tf).vector_field
        jfield = jax.jit(driver.JaxIntegrator(jh, jf).vector_field)
        s = protocol["settings"]
        dt = s["coarse_output_step_M"]
        result["stages"]["initial"] = {
            "torch_state": array(ts).tolist(), "jax_state": array(js).tolist(),
            "state_max_abs": float(np.max(np.abs(array(ts)-array(js)))),
            "same_state_field_max_abs": float(np.max(np.abs(
                array(tfield(ts))-array(jfield(jnp.asarray(array(ts))))))),
            "first_rk4_step_M": dt,
            "first_rk4_state_max_abs": float(np.max(np.abs(array(rk4(tfield, ts, dt))
                - array(rk4(jfield, js, dt))))),
        }
        save()
        tc = integrate_adaptive(tfield, ts, dt, t_end=s["max_duration_M"],
            event=AlignedSpinV4Event("coarse"), initial_step=dt,
            max_step=s["max_step_M"], controller="gsl_state_scaled",
            atol=s["atol"], rtol=s["rtol"])
        print(case_id, "Torch coarse done", flush=True)
        jrun = integrate_two_stage_trajectory(jfield, js, output_step=dt,
            max_duration=s["max_duration_M"], coarse_initial_step=dt,
            coarse_max_step=s["max_step_M"], fine_max_step=s["max_step_M"],
            fine_refinement_factor=s["fine_refinement_factor"],
            atol=s["atol"], rtol=s["rtol"])
        print(case_id, "JAX coarse/fine done", flush=True)
        restart = (len(tc.output_time)-math.ceil(150/dt)
                   if (len(tc.output_time)-1)*dt >= 150
                   else len(tc.output_time)-math.ceil(len(tc.output_time)/2))
        fine_dt = dt/s["fine_refinement_factor"]
        tfr = integrate_adaptive(tfield, tc.output_state[restart], fine_dt,
            t_end=s["max_duration_M"], event=AlignedSpinV4Event("fine"),
            initial_step=fine_dt, max_step=s["max_step_M"],
            controller="gsl_state_scaled", atol=s["atol"], rtol=s["rtol"])
        print(case_id, "Torch fine done", flush=True)
        ct, cs, ft, fs = tc.output_time, tc.output_state, tfr.output_time, tfr.output_state
        jct, jcs, jft, jfs = jrun.coarse_time, jrun.coarse_state, jrun.fine_time, jrun.fine_state
        result["stages"]["trajectory"] = {
            "torch_restart_index": restart, "jax_restart_index": jrun.restart_index,
            "torch_coarse_samples": len(ct), "jax_coarse_samples": len(jct),
            "torch_fine_samples": len(ft), "jax_fine_samples": len(jft),
            "coarse_time": compare(jct, ct), "coarse_state": compare(jcs, cs),
            "fine_time": compare(jft, ft), "fine_state": compare(jfs, fs),
            "torch_coarse_event_time_M": float(tc.time[-1]),
            "jax_coarse_event_time_M": float(array(jrun.coarse.time)[-1]),
            "torch_fine_event_time_M": float(tfr.time[-1]),
            "jax_fine_event_time_M": float(array(jrun.fine.time)[-1]),
        }
        save()
        mode_options = driver.MODE_OPTIONS | {"retain_pan_cubic_spin": True}
        two = tfield(fs)[:, 1]
        jfo = jax.vmap(jfield)(jfs)[:, 1]
        tco = tfield(cs)[:, 1]
        jco = jax.vmap(jfield)(jcs)[:, 1]
        traw, trf = driver.torch_mode(fs, th, tf, **mode_options)
        jmode_fn = jax.jit(lambda state: driver.jax_mode(state, jh, jf, **mode_options))
        jraw, jrf = jmode_fn(jfs)
        jsame, _ = jmode_fn(jnp.asarray(array(fs)))
        tcraw, _ = driver.torch_mode(cs, th, tf, **mode_options)
        jcraw, _ = jmode_fn(jcs)
        result["stages"]["raw"] = {
            "fine_same_states": compare(jsame, traw),
            "fine_own_states": compare(jraw, traw),
            "coarse_own_states": compare(jcraw, tcraw),
            "fine_frequency_own": compare(jrf, trf),
        }
        save()
        tnu, tchi = driver.torch_parameters(*map(tensor, (m1,m2,chi1,chi2)))
        jnu, jchi = driver.jax_parameters(m1,m2,chi1,chi2)
        peak = ta._full_orbital_peak(ft, two)
        match = peak.time + driver.torch_offset(tnu, tchi)
        phase_capture, rhs_capture = [], []
        phase_matrix = _phase_matrix(ft, fs, two, match,
            half_width_M=0.0, degree=0, field=tfield, capture=phase_capture)
        center = phase_capture[0]["interpolated_state"]
        amplitude_rows, amplitude_raw = _continuous_amplitude(center, th, tf, tfield)
        rhs = _rhs_derivatives(ft, traw, match, center_state=center,
            hamiltonian=th, flux=tf, field=tfield, capture=rhs_capture)
        rhs[0, :] = amplitude_rows[1:, 0]
        amplitude_derivatives = amplitude_rows[1:, 1:4]
        targets = torch_targets(tnu, tchi)
        amplitude_matrix = amplitude_rows[:, 1:4]
        amplitude_rhs = torch.stack(targets[:3])-amplitude_rows[:, 0]
        phase_rhs = rhs[1, :] - torch.stack(targets[3:])
        tmass = driver.torch_final_mass(*map(tensor, (m1,m2,chi1,chi2)))
        jmass = driver.jax_final_mass(m1,m2,chi1,chi2)
        tspin = driver.torch_final_spin(*map(tensor,(m1,m2,chi1,chi2)), fit_order=(3,4))
        jspin = driver.jax_final_spin(m1,m2,chi1,chi2,fit_order=(3,4))
        tw, tdamp = driver.torch_qnm(tspin)
        jw, jdamp = driver.jax_qnm(jspin)
        tsigma = torch.complex(tw,-tdamp)/tmass
        jsigma = (jw-1j*jdamp)/jmass
        fine_count = math.ceil((float(match)+100.0)/fine_dt)+1
        output_time = torch.arange(fine_count,dtype=torch.float64)*fine_dt
        controls = {"amplitude_matrix": amplitude_matrix, "amplitude_rhs": amplitude_rhs,
            "phase_matrix": phase_matrix, "phase_rhs": phase_rhs, "match_time": match}
        result["stages"]["controls"] = {
            "torch_peak_time_M": float(peak.time), "match_time_M": float(match),
            "amplitude_rows": array(amplitude_rows).tolist(),
            "amplitude_raw": amplitude_raw,
            "amplitude_matrix": array(amplitude_matrix).tolist(),
            "amplitude_rhs": array(amplitude_rhs).tolist(),
            "phase_matrix": array(phase_matrix).tolist(),
            "phase_rhs": array(phase_rhs).tolist(),
            "target_values": [float(value) for value in targets],
            "phase_capture": phase_capture[0], "rhs_capture": rhs_capture[0],
            "torch_control_sha256_before_frequency": control_sha(controls),
        }
        save()
        original_tfull, original_teval = ta._full_series_nqc, ta._eval_batch_scalar
        first_hook = []
        with patch.object(ta, "_full_series_nqc", _amplitude_hook(
                original_tfull, original_teval, amplitude_rows, first_hook)):
            tnatural = driver.torch_attach(ft,fs,two,traw,trf,tnu,tchi,tsigma,
                output_time,match_time=match,
                diagnostic_amplitude_derivatives=amplitude_derivatives,
                diagnostic_phase_matrix=phase_matrix,
                diagnostic_matching_rhs_derivatives=rhs,
                diagnostic_continuous_boundary=True)
        if len(first_hook) != 1 or not np.array_equal(
                np.asarray(first_hook[0]["amplitude_rhs"]), array(amplitude_rhs)):
            raise ValueError("Torch natural attachment did not use exact amplitude RHS")
        with torch.enable_grad():
            center_state = tensor(center).detach().requires_grad_(True)
            center_vector = tfield(center_state)
            _, raw_frequency = driver.torch_mode(center_state, th, tf, **mode_options)
            phase_basis = driver.torch_basis(center_state[0], center_state[2],
                center_vector[1])[1]
            basis_first = torch.stack(tuple(
                (torch.autograd.grad(phase_basis[i],center_state,retain_graph=True)[0]
                 *center_vector).sum() for i in range(2)))
            ode_frequency = float((raw_frequency - basis_first.dot(
                tnatural.calibration.coefficients.phase.detach())).detach())
        controls["ode_boundary_frequency"] = np.asarray(ode_frequency,dtype=np.float64)
        jcontrols = {name: jnp.asarray(array(value)) for name,value in controls.items()}
        tsha, jsha = control_sha(controls), control_sha(jcontrols)
        hooks = {"torch_amplitude":[], "jax_amplitude":[], "ringdown":{}}
        original_jcal = ja.calibrate_nqc_22_full_series
        original_tring, original_jring = ta.merger_ringdown_22, ja.merger_ringdown_22

        def jcal(*args, **kwargs):
            nqc = original_jcal(*args, **kwargs)
            am, ar = jcontrols["amplitude_matrix"], jcontrols["amplitude_rhs"]
            pm, pr = jcontrols["phase_matrix"], jcontrols["phase_rhs"]
            a, ac, ax = ja._solve_checked(am,ar,kwargs.get("max_condition",1e10),
                "matched ODE amplitude")
            b, pc, px = ja._solve_checked(pm,pr,kwargs.get("max_condition",1e10),
                "matched ODE phase")
            cal = nqc.calibration._replace(
                coefficients=nqc.calibration.coefficients._replace(amplitude=a,phase=b),
                amplitude_condition=ac,amplitude_residual=ax,
                phase_condition=pc,phase_residual=px)
            corrected_amplitude = nqc.sampled_amplitude + jnp.sum(
                nqc.sampled_products*a,axis=-1)
            if not bool(jnp.all(corrected_amplitude>0)):
                raise ValueError("nonpositive matched corrected amplitude")
            corrected_mode = corrected_amplitude*jnp.exp(-1j*(
                nqc.sampled_waveform_phase-jnp.sum(nqc.sampled_phase_basis*b,axis=-1)))
            hooks["jax_amplitude"].append({
                "natural_amplitude_coefficients": array(nqc.coefficients.amplitude).tolist(),
                "natural_phase_coefficients": array(nqc.coefficients.phase).tolist(),
                "matched_amplitude_coefficients": array(a).tolist(),
                "matched_phase_coefficients": array(b).tolist(),
            })
            return replace(nqc,calibration=cal,
                amplitude_matching_matrix=am,phase_matching_matrix=pm,
                sampled_corrected_amplitude=corrected_amplitude,
                sampled_corrected_phase=ja.unwrap_mode_phase(corrected_mode))

        def tring(elapsed,sigma,amplitude,amplitude_derivative,frequency,phase,shape):
            hooks["ringdown"]["torch_natural_frequency"] = float(frequency)
            hooks["ringdown"]["torch_injected_frequency"] = ode_frequency
            return original_tring(elapsed,sigma,amplitude,amplitude_derivative,
                tensor(ode_frequency),phase,shape)

        def jring(elapsed,sigma,amplitude,amplitude_derivative,frequency,phase,shape):
            hooks["ringdown"]["jax_natural_frequency"] = float(frequency)
            hooks["ringdown"]["jax_injected_frequency"] = ode_frequency
            return original_jring(elapsed,sigma,amplitude,amplitude_derivative,
                jnp.asarray(ode_frequency),phase,shape)

        with (patch.object(ta,"_full_series_nqc",_amplitude_hook(
                  original_tfull,original_teval,amplitude_rows,hooks["torch_amplitude"])),
              patch.object(ta,"merger_ringdown_22",tring)):
            tref = driver.torch_attach(ft,fs,two,traw,trf,tnu,tchi,tsigma,
                output_time,match_time=match,
                diagnostic_amplitude_derivatives=amplitude_derivatives,
                diagnostic_phase_matrix=phase_matrix,
                diagnostic_matching_rhs_derivatives=rhs,
                diagnostic_continuous_boundary=True)
        with (patch.object(ja,"calibrate_nqc_22_full_series",jcal),
              patch.object(ja,"merger_ringdown_22",jring)):
            jref = driver.jax_attach(jft,jfs,jfo,jraw,jrf,jnu,jchi,jsigma,
                jnp.asarray(array(output_time)),match_time=jcontrols["match_time"],
                diagnostic_amplitude_derivatives=jnp.asarray(array(amplitude_derivatives)),
                diagnostic_phase_matrix=jcontrols["phase_matrix"],
                diagnostic_matching_rhs_derivatives=jnp.asarray(array(rhs)),
                diagnostic_continuous_boundary=True)
        tcoef,jcoef=tref.calibration.coefficients,jref.calibration.coefficients
        tmode,jmode=array(tref.mode),array(jref.mode)
        before=array(output_time)<float(match)
        after=~before
        result["stages"]["attachment"] = {
            "torch_control_sha256":tsha,"jax_control_sha256":jsha,
            "ode_boundary_frequency":ode_frequency,
            "torch_boundary_index":int(tref.boundary_index),
            "jax_boundary_index":int(jref.boundary_index),
            "boundary_time_abs_M":abs(float(tref.boundary_time)-float(jref.boundary_time)),
            "amplitude_coeff_max_abs":float(np.max(np.abs(array(tcoef.amplitude)-array(jcoef.amplitude)))),
            "phase_coeff_max_abs":float(np.max(np.abs(array(tcoef.phase)-array(jcoef.phase)))),
            "corrected_inspiral":compare(jref.corrected_inspiral,tref.corrected_inspiral),
            "pre_boundary_mode":compare(jmode[before],tmode[before]),
            "post_boundary_mode":compare(jmode[after],tmode[after]),
            "fine_whole_mode":compare(jmode,tmode),
            "torch_natural_boundary_frequency":float(tnatural.boundary_frequency),
            "torch_natural_boundary_amplitude":float(tnatural.boundary_amplitude),
            "jax_patched_boundary_amplitude":float(jref.boundary_amplitude),
            "torch_patched_boundary_amplitude":float(tref.boundary_amplitude),
            "hooks":hooks,
        }
        save()
        tnqc=driver.torch_factor(cs[:,0],cs[:,2],tco,tcoef)
        jnqc=driver.jax_factor(jcs[:,0],jcs[:,2],jco,jcoef)
        tprefix=array(tcraw[:restart]*tnqc[:restart])
        jprefix=array(jcraw[:jrun.restart_index]*jnqc[:jrun.restart_index])
        refinement=s["fine_refinement_factor"]
        tfull=np.concatenate((tprefix,tmode[::refinement]))
        jfull=np.concatenate((jprefix,jmode[::refinement]))
        ttime=np.concatenate((array(ct[:restart]),float(ct[restart])+array(output_time)[::refinement]))
        jtime=np.concatenate((array(jct[:jrun.restart_index]),
            float(jct[jrun.restart_index])+array(output_time)[::refinement]))
        tcrop=final_h22_crop(ttime,tfull,requested)
        jcrop=final_h22_crop(jtime,jfull,requested)
        result["stages"]["whole"]={
            "torch_samples":len(tfull),"jax_samples":len(jfull),
            "time":compare(jtime,ttime),"coarse_prefix":compare(jprefix,tprefix),
            "assembled":compare(jfull,tfull),
            "common_after_torch_crop":compare(jfull[tcrop["index"]:],tfull[tcrop["index"]:]),
            "torch_crop":tcrop,"jax_crop":jcrop,
            "crop_index_difference":jcrop["index"]-tcrop["index"],
        }
        save()
        g=protocol["gates"]
        st=result["stages"]
        ini,tr,raw,at,wh=(st[name] for name in ("initial","trajectory","raw","attachment","whole"))
        result["checks"]={
            "input_sha_replay":g["input_sha_replay"],
            "independent_initial_state":ini["state_max_abs"]<=g["initial_state_max_abs"],
            "same_state_initial_field":ini["same_state_field_max_abs"]<=g["initial_field_same_state_max_abs"],
            "first_fixed_step":ini["first_rk4_state_max_abs"]<=g["first_fixed_step_state_max_abs"],
            "restart_index":tr["torch_restart_index"]==tr["jax_restart_index"]
                and g["restart_index_difference"]==0,
            "coarse_time":tr["coarse_time"]["max_abs"]<=g["coarse_fine_time_max_abs_M"]
                and tr["coarse_time"]["left_samples"]==tr["coarse_time"]["right_samples"],
            "fine_time":tr["fine_time"]["max_abs"]<=g["coarse_fine_time_max_abs_M"]
                and tr["fine_time"]["left_samples"]==tr["fine_time"]["right_samples"],
            "same_state_raw":raw["fine_same_states"]["relative_l2"]<=g["same_state_raw_l2"],
            "own_state_raw":raw["fine_own_states"]["relative_l2"]<=g["own_state_raw_l2"],
            "exact_control_sha":at["torch_control_sha256"]==at["jax_control_sha256"],
            "amplitude_coefficients":at["amplitude_coeff_max_abs"]<=g["nqc_amplitude_coefficient_max_abs"],
            "phase_coefficients":at["phase_coeff_max_abs"]<=g["nqc_phase_coefficient_max_abs"],
            "boundary_time":at["boundary_time_abs_M"]<=g["attachment_boundary_time_max_abs_M"]
                and at["torch_boundary_index"]==at["jax_boundary_index"],
            "identical_injected_frequency":at["hooks"]["ringdown"]["torch_injected_frequency"]
                ==at["hooks"]["ringdown"]["jax_injected_frequency"],
            "corrected_inspiral":at["corrected_inspiral"]["relative_l2"]<=g["corrected_inspiral_l2"],
            "pre_boundary":at["pre_boundary_mode"]["relative_l2"]<=g["pre_boundary_mode_l2"],
            "post_boundary":at["post_boundary_mode"]["relative_l2"]<=g["post_boundary_mode_l2"],
            "coarse_prefix":wh["coarse_prefix"]["relative_l2"]<=g["coarse_prefix_l2"],
            "assembled_whole":wh["assembled"]["relative_l2"]<=g["whole_mode_l2"],
            "cropped_whole":wh["common_after_torch_crop"]["relative_l2"]<=g["cropped_whole_mode_l2"],
            "whole_chart":wh["time"]["max_abs"]<=g["coarse_fine_time_max_abs_M"]
                and wh["torch_samples"]==wh["jax_samples"],
            "crop_index":wh["crop_index_difference"]==g["independent_crop_index_difference"],
            "strict_crops":wh["torch_crop"]["strict_first_interior_crossing"]
                and wh["jax_crop"]["strict_first_interior_crossing"],
            "nonzero_series":g["finite_nonzero_comparison_series"],
            "hooks_reached":len(hooks["torch_amplitude"])==len(hooks["jax_amplitude"])==1
                and set(hooks["ringdown"])=={
                    "torch_natural_frequency","jax_natural_frequency",
                    "torch_injected_frequency","jax_injected_frequency"},
        }
        result["diagnostic_checks_passed"]=all(result["checks"].values())
        result["status"]="diagnostic_complete" if result["diagnostic_checks_passed"] else "failed_parity"
        result["elapsed_s"]=time.monotonic()-started
        save()
    except BaseException as error:
        result.update({"status":"failed_execution","release_qualified":False,
            "error":repr(error),"traceback":traceback.format_exc(),
            "elapsed_s":time.monotonic()-started})
        save()
        raise


def supervise(case_id):
    """Bound one row in a fresh process; keep its latest atomic stage result."""
    protocol, execution, _ = checked_inputs(case_id)
    output = EVIDENCE / execution["outputs"][case_id]
    if output.exists():
        raise FileExistsError(f"evidence already exists: {output}")
    limits = protocol["execution_bounds"]
    if (limits != execution["watchdog"]
            or not limits["sequential_rows_only"]):
        raise ValueError("watchdog differs from frozen protocol")
    command = [sys.executable, "-m", "dev.native_rom.diagnose_warm_jax_two_row_domain_v1",
               "--case", case_id, "--worker"]
    child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    print(case_id, "worker PID", child.pid, flush=True)
    started = time.monotonic()
    reason = None
    while child.poll() is None:
        elapsed = time.monotonic() - started
        if elapsed > limits["wall_seconds_per_row"]:
            reason = f"wall_seconds>{limits['wall_seconds_per_row']}"
        else:
            sample = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(child.pid)],
                capture_output=True, text=True, check=False)
            if sample.returncode == 0 and sample.stdout.strip():
                rss_kib = int(sample.stdout.strip())
                if rss_kib > limits["rss_gib_stop_threshold"] * 1024**2:
                    reason = f"rss_kib>{limits['rss_gib_stop_threshold']}GiB"
        if reason is not None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=limits["term_to_kill_grace_seconds"])
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            partial = json.loads(output.read_text()) if output.exists() else {
                "schema": "diffgw.seobnrv4.warm-jax-two-row-domain.v1",
                "case_id": case_id, "stages": {}, "checks": {}}
            partial.update({"status": "failed_watchdog", "release_qualified": False,
                "watchdog_reason": reason, "worker_pid": child.pid,
                "worker_exit_code": child.returncode,
                "elapsed_s": time.monotonic() - started})
            temporary = output.with_suffix(".tmp")
            temporary.write_text(json.dumps(partial, indent=2, sort_keys=True,
                allow_nan=False) + "\n")
            temporary.replace(output)
            return 124
        time.sleep(0.5)
    if child.returncode != 0 and output.exists():
        partial = json.loads(output.read_text())
        if partial.get("status") == "running":
            partial.update({"status": "failed_process", "release_qualified": False,
                "worker_pid": child.pid, "worker_exit_code": child.returncode,
                "elapsed_s": time.monotonic() - started})
            temporary = output.with_suffix(".tmp")
            temporary.write_text(json.dumps(partial, indent=2, sort_keys=True,
                allow_nan=False) + "\n")
            temporary.replace(output)
    return child.returncode


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--case",required=True,choices=("q4_nonspinning","strong_antialigned"))
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        run(args.case)
    else:
        raise SystemExit(supervise(args.case))
