"""Replay cached TEOB NQC grids under the PA energy-scaled acceleration policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from torchwave.approximants.eob.teob_conditioned_waveform_clean import (
    _make_hamiltonian,
    project_teob22_generic,
    _stage_aware_frozen_mobility_radial_acceleration,
)
from torchwave.approximants.eob.teob_merger_nqc_clean import nqc_basis_values
from torchwave.approximants.eob.teob_ringdown_assembly_clean import assemble_teob_ringdown_22
from torchwave.approximants.eob.td_fd_conditioning_clean import condition_td_to_fd, plan_td_support
from torchwave.approximants.eob.teob_uib2016_remnant import uib2016_nonspinning_remnant
from types import SimpleNamespace


def _fd_replay(data, amplitude, phase, aux_amplitude, aux_phase, mass1, mass2, f_low):
    dtype = torch.float64
    tm = torch.as_tensor(data["total_mass_seconds"], dtype=dtype)
    final_mass, final_spin = uib2016_nonspinning_remnant(
        torch.tensor(mass1, dtype=dtype), torch.tensor(mass2, dtype=dtype),
    )
    ringdown = assemble_teob_ringdown_22(
        torch.as_tensor(data["continuation_time"], dtype=dtype), amplitude, phase,
        torch.as_tensor(data["nqc_auxiliary_time"], dtype=dtype), aux_amplitude,
        aux_phase, torch.as_tensor(data["nqc_auxiliary_orbital_frequency"], dtype=dtype),
        torch.as_tensor(mass1 * mass2 / (mass1 + mass2) ** 2, dtype=dtype),
        final_mass, final_spin, output_step=torch.as_tensor(data["sampling_dt_seconds"], dtype=dtype) / tm,
    )
    hp, hc = project_teob22_generic(
        ringdown.mode, torch.conj(ringdown.mode), mass1=mass1, mass2=mass2,
        distance=100.0, inclination=0.0, phic=0.0,
    )
    support = plan_td_support(torch.tensor(mass1, dtype=dtype), torch.tensor(mass2, dtype=dtype),
                              torch.tensor(f_low, dtype=dtype), dtype=dtype)
    return condition_td_to_fd(
        hp, hc, dt=torch.as_tensor(data["sampling_dt_seconds"], dtype=dtype),
        delta_f=torch.as_tensor(data["sampling_delta_f"], dtype=dtype),
        f_max=torch.as_tensor(data["sampling_f_nyquist"], dtype=dtype), support=support,
        tc=0.0, return_metadata=True,
    )


def replay(path: Path, mass1: float, mass2: float, public: dict[str, np.ndarray], f_low: float) -> dict[str, object]:
    data = np.load(path)
    dtype = torch.float64
    ham = _make_hamiltonian(torch.tensor(mass1, dtype=dtype), torch.tensor(mass2, dtype=dtype))
    state = torch.as_tensor(data["continuation_state"], dtype=dtype)
    continuation = SimpleNamespace(
        state=state,
        pa_time=torch.as_tensor(data["pa_time"], dtype=dtype),
        pa_state=state[: data["pa_time"].size],
    )
    baseline = torch.as_tensor(data["radial_acceleration"], dtype=dtype)
    stage = _stage_aware_frozen_mobility_radial_acceleration(ham, continuation)
    n_pa = continuation.pa_time.numel()
    # Carry both NQC variants through the cached ringdown and conditioning.
    # This does not evolve an ODE; the cache supplies every trajectory knot.
    # Use the cached baseline solve coefficients, which makes the baseline
    # reconstruction an exact replay.  The auxiliary grid is entirely ODE,
    # so its stage-aware series is unchanged; only full-grid PA samples need
    # a new basis evaluation.
    amp = torch.as_tensor(data["nqc_original_amplitude"], dtype=dtype)
    phase = torch.as_tensor(data["nqc_original_phase"], dtype=dtype)
    aux_amp = torch.as_tensor(data["nqc_auxiliary_corrected_amplitude"], dtype=dtype)
    aux_phase = torch.as_tensor(data["nqc_auxiliary_corrected_phase"], dtype=dtype)
    coeff_a = torch.as_tensor(data["nqc_amplitude_coefficients"], dtype=dtype)
    coeff_p = torch.as_tensor(data["nqc_phase_coefficients"], dtype=dtype)
    stage_basis = nqc_basis_values(state[:, 0], state[:, 2],
                                   torch.as_tensor(data["continuation_frequency"], dtype=dtype), stage)
    stage_amp = amp * (1.0 + (stage_basis.amplitude * coeff_a).sum(-1))
    stage_phase = phase - (stage_basis.phase * coeff_p).sum(-1)
    # The cache stores the canonical solved series; use it directly so this
    # first stage is an exact assembly replay independent of any solver
    # re-evaluation or printed target precision.
    baseline_amp = torch.as_tensor(data["nqc_original_corrected_amplitude"], dtype=dtype)
    baseline_phase = torch.as_tensor(data["nqc_original_corrected_phase"], dtype=dtype)
    def l2(a, b):
        return float((torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).detach())
    old_fd = _fd_replay(data, baseline_amp, baseline_phase, aux_amp, aux_phase, mass1, mass2, f_low)
    new_fd = _fd_replay(data, stage_amp, stage_phase, aux_amp, aux_phase, mass1, mass2, f_low)
    baseline_plus_l2 = l2(old_fd.h_plus, torch.as_tensor(data["h_plus_fd"]))
    baseline_cross_l2 = l2(old_fd.h_cross, torch.as_tensor(data["h_cross_fd"]))
    if max(baseline_plus_l2, baseline_cross_l2) > 1.0e-12:
        raise RuntimeError(f"cached baseline replay failed: plus={baseline_plus_l2:.3e}, cross={baseline_cross_l2:.3e}")
    pub_plus = torch.as_tensor(public["public_h_plus_fd"])
    pub_cross = torch.as_tensor(public["public_h_cross_fd"])
    if old_fd.frequencies.shape != torch.as_tensor(public["public_frequencies"]).shape or not torch.equal(
        old_fd.frequencies, torch.as_tensor(public["public_frequencies"])
    ):
        raise RuntimeError("cached and public frequency axes disagree")
    catalog_hi = 256.0 if mass1 > 10 else 500.0
    catalog = (public["public_frequencies"] >= f_low) & (public["public_frequencies"] <= catalog_hi)
    full = torch.ones_like(pub_plus, dtype=torch.bool)
    suffix = slice(n_pa, None)
    return {
        "cache": str(path),
        "pa_knots": int(n_pa),
        "total_knots": int(state.shape[0]),
        "max_pa_energy_factor_minus_one": float((stage[:n_pa] / baseline[:n_pa] - 1).abs().max().detach()),
        "max_ode_acceleration_difference": float((stage[suffix] - baseline[suffix]).abs().max().detach()),
        "nqc_coefficients": "cached canonical solve held fixed for both variants",
        "max_original_corrected_amplitude_difference": float((stage_amp - baseline_amp).abs().max().detach()),
        "max_original_corrected_phase_difference": float((stage_phase - baseline_phase).abs().max().detach()),
        "max_aux_corrected_amplitude_difference": 0.0,
        "max_aux_corrected_phase_difference": 0.0,
        "baseline_native_plus_raw_l2": baseline_plus_l2,
        "baseline_native_cross_raw_l2": baseline_cross_l2,
        "catalog_frequency_band_hz": [f_low, catalog_hi],
        "stage_vs_baseline_plus_raw_l2_full": l2(new_fd.h_plus, old_fd.h_plus),
        "stage_vs_baseline_cross_raw_l2_full": l2(new_fd.h_cross, old_fd.h_cross),
        "stage_vs_public_plus_raw_l2_full": l2(new_fd.h_plus[full], pub_plus[full]),
        "stage_vs_public_cross_raw_l2_full": l2(new_fd.h_cross[full], pub_cross[full]),
        "stage_vs_public_combined_raw_l2_full": l2(torch.cat((new_fd.h_plus[full], new_fd.h_cross[full])), torch.cat((pub_plus[full], pub_cross[full]))),
        "stage_vs_public_plus_raw_l2_catalog": l2(new_fd.h_plus[catalog], pub_plus[catalog]),
        "stage_vs_public_cross_raw_l2_catalog": l2(new_fd.h_cross[catalog], pub_cross[catalog]),
        "stage_vs_public_combined_raw_l2_catalog": l2(torch.cat((new_fd.h_plus[catalog], new_fd.h_cross[catalog])), torch.cat((pub_plus[catalog], pub_cross[catalog]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    args = parser.parse_args()
    source_paths = [
        Path("src/torchwave/approximants/eob/teob_conditioned_waveform_clean.py"),
        Path("src/torchwave/approximants/eob/teob_merger_nqc_clean.py"),
        Path("src/torchwave/approximants/eob/teob_ringdown_assembly_clean.py"),
        Path("src/torchwave/approximants/eob/td_fd_conditioning_clean.py"),
        Path(__file__),
    ]
    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    source_before = {str(p): sha256(p) for p in source_paths}
    input_paths = [args.native, args.catalog, args.public]
    input_before = {str(p): sha256(p) for p in input_paths}
    runtime_before = {"python_executable": sys.executable, "python_sha256": sha256(Path(sys.executable)),
                      "python_version": sys.version, "torch_version": torch.__version__,
                      "platform": platform.platform()}
    started = time.perf_counter()
    report = {
        "policy": "frozen-mobility-stage-aware",
        "convention": "H_real(r,p_r*,p_phi)^2 applies only to continuation PA prefix; ODE suffix is unchanged",
        "cases": [
            replay(args.native, 30.0, 20.0, {
                "public_frequencies": np.load(args.public)["30p20_public_frequencies"],
                "public_h_plus_fd": np.load(args.public)["30p20_public_h_plus_fd"],
                "public_h_cross_fd": np.load(args.public)["30p20_public_h_cross_fd"],
            }, 20.0),
            replay(args.catalog, 1.4, 1.4, {
                "public_frequencies": np.load(args.public)["1_4p1_4_public_frequencies"],
                "public_h_plus_fd": np.load(args.public)["1_4p1_4_public_h_plus_fd"],
                "public_h_cross_fd": np.load(args.public)["1_4p1_4_public_h_cross_fd"],
            }, 30.0),
        ],
    }
    report["source_sha256_before"] = source_before
    report["source_sha256_after"] = {str(p): sha256(p) for p in source_paths}
    report["input_npz_sha256_before"] = input_before
    report["input_npz_sha256_after"] = {str(p): sha256(p) for p in input_paths}
    report["runtime_sha256_before"] = runtime_before
    report["runtime_sha256_after"] = {"python_executable": sys.executable, "python_sha256": sha256(Path(sys.executable)),
                                       "python_version": sys.version, "torch_version": torch.__version__,
                                       "platform": platform.platform()}
    report["hashes_unchanged"] = (report["source_sha256_before"] == report["source_sha256_after"]
                                   and report["input_npz_sha256_before"] == report["input_npz_sha256_after"]
                                   and report["runtime_sha256_before"] == report["runtime_sha256_after"])
    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
