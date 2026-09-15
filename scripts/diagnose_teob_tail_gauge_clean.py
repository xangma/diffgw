#!/usr/bin/env python3
"""Run the bounded public-fixture regression for the TEOB tail gauge."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from torchwave.approximants.eob.teob_radiation_clean import (
    TEOBRadiationClean,
    teob_tail_gauge_factor,
)
from torchwave.approximants.eob.teob_ringdown_assembly_clean import (
    natural_cubic_interpolate,
)
from torchwave.approximants.teobresums import TEOBResumSHamiltonian
from torchwave.constants import C_SI, MPC_SI


FIXTURE = Path("/tmp/torchwave-teob-pa-public-gauge.npz")
FIXTURE_SHA256 = "c81bcb89086833fa6c44fc44a26eb54ee93adc2ff14fca5976944aaae54c8fb0"
DEFAULT_REPORT = Path(
    "docs/reviews/2026-09-13-parity-review/verification/teob-tail-gauge-clean/diagnostic.json"
)
SOURCE_FILES = (
    Path("src/torchwave/approximants/eob/teob_radiation_clean.py"),
    Path("src/torchwave/approximants/eob/factorized_modes.py"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _float_tensor(data: np.lib.npyio.NpzFile, key: str) -> torch.Tensor:
    return torch.from_numpy(data[key]).to(torch.float64)


def _mode_and_inputs(data: np.lib.npyio.NpzFile) -> tuple[torch.Tensor, TEOBRadiationClean, dict[str, torch.Tensor], TEOBResumSHamiltonian]:
    state = _float_tensor(data, "pa_state")
    radius, phase, prstar, pphi = state.unbind(-1)
    omega = _float_tensor(data, "Omega")
    zero = torch.tensor(0.0, dtype=torch.float64)
    hamiltonian = TEOBResumSHamiltonian(
        torch.tensor(0.24, dtype=torch.float64), zero, zero, zero, zero,
        delta=torch.tensor(0.2, dtype=torch.float64),
    )
    heff, hreal, hcirc, aprime = hamiltonian.clean_radiation_inputs(
        radius, prstar, pphi,
    )
    inputs = {
        "radius": radius,
        "omega": omega,
        "phase": phase,
        "prstar": prstar,
        "pphi": pphi,
        "heff": heff,
        "hreal": hreal,
        "hcirc": hcirc,
        "aprime": aprime,
    }
    radiation = TEOBRadiationClean(8, residual_phase="zero-unpublished")
    mode = radiation.mode(
        2, 2, radius, omega, phase, prstar, pphi, heff, hreal, hcirc,
        aprime, hamiltonian.m1, hamiltonian.m2,
    )
    return mode, radiation, inputs, hamiltonian


def _run(report_path: Path) -> dict[str, object]:
    started = datetime.now(timezone.utc).isoformat()
    before = {str(path): _sha256(path) for path in SOURCE_FILES}
    if not FIXTURE.is_file():
        raise FileNotFoundError(f"public fixture is unavailable: {FIXTURE}")
    fixture_sha = _sha256(FIXTURE)
    if fixture_sha != FIXTURE_SHA256:
        raise RuntimeError(f"fixture hash changed: expected {FIXTURE_SHA256}, got {fixture_sha}")

    with np.load(FIXTURE, allow_pickle=False) as data:
        mode, radiation, inputs, hamiltonian = _mode_and_inputs(data)
        mode_before = torch.from_numpy(data["mode_before"])
        hreal_fixture = _float_tensor(data, "Hreal")
        omega = inputs["omega"]
        gauge = teob_tail_gauge_factor(2, hreal_fixture, omega)
        expected = mode_before * gauge
        amplitudes = radiation.amplitudes(
            inputs["radius"], omega, inputs["phase"], inputs["prstar"], inputs["pphi"],
            inputs["heff"], inputs["hreal"], inputs["hcirc"], inputs["aprime"],
            hamiltonian.m1, hamiltonian.m2,
        )
        infinity_flux = radiation.flux(
            inputs["radius"], omega, inputs["phase"], inputs["prstar"], inputs["pphi"],
            inputs["heff"], inputs["hreal"], inputs["hcirc"], inputs["aprime"],
            hamiltonian.m1, hamiltonian.m2,
        )
        amplitude_flux = sum(
            m * m * omega * value.square() for (_, m), value in amplitudes.items()
        ) / (8.0 * math.pi)

        public_time = torch.from_numpy(data["public_time"]).to(torch.float64)
        pa_time = torch.from_numpy(data["pa_time"]).to(torch.float64)
        pa_state = torch.from_numpy(data["pa_state"]).to(torch.float64)
        orbital_phase = pa_state[:, 1]
        valid = (public_time >= pa_time[0]) & (public_time <= pa_time[-1])
        query = public_time[valid]
        # Remove the orbital carrier before unwrapping the sparse PA residual,
        # then restore it as a positive phase.  Interpolate amplitude and phase
        # independently onto native dense public times; never interpolate the
        # public complex strain onto PA knots.
        phase_knots = 2.0 * orbital_phase - torch.from_numpy(
            np.unwrap(np.angle(
                mode.detach().numpy() * np.exp(2j * orbital_phase.numpy())
            ))
        ).to(torch.float64)
        amplitude = natural_cubic_interpolate(pa_time, mode.abs(), query)
        phase = natural_cubic_interpolate(pa_time, phase_knots, query)
        mode_dense = torch.polar(amplitude, -phase).detach().numpy()
        public = (
            data["public_hp"][valid.numpy()]
            - 1j * data["public_hc"][valid.numpy()]
        )
        scale = -math.sqrt(5.0 / (4.0 * math.pi)) * float(data["Msec"]) * C_SI / (100.0 * MPC_SI)
        projected = scale * mode_dense
        ratio = public / projected
        phase_residual = np.unwrap(np.angle(ratio))
        relative_l2 = np.linalg.norm(public - projected) / np.linalg.norm(public)
        public_amplitude = np.abs(public)
        projected_amplitude = np.abs(projected)
        amplitude_relative_l2 = (
            np.linalg.norm(projected_amplitude - public_amplitude)
            / np.linalg.norm(public_amplitude)
        )
        amplitude_relative_max_abs = np.max(
            np.abs(projected_amplitude / public_amplitude - 1.0)
        )

        after = {str(path): _sha256(path) for path in SOURCE_FILES}
        result: dict[str, object] = {
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "cwd": str(Path.cwd()),
            "command": " ".join((sys.executable, *sys.argv)),
            "python_executable": sys.executable,
            "pythonpath": os.environ.get("PYTHONPATH", ""),
            "pid": os.getpid(),
            "log_path": str(report_path.with_suffix(".log")),
            "bounded": True,
            "sample_count": int(mode.numel()),
            "valid_public_projection_samples": int(valid.sum()),
            "public_time_samples": int(public_time.numel()),
            "public_time_native_spacing": float(public_time[1] - public_time[0]),
            "pa_time_knots": int(pa_time.numel()),
            "pa_time_support": [float(pa_time[0]), float(pa_time[-1])],
            "public_time_support_used": [float(query[0]), float(query[-1])],
            "public_comparison_sampling": {
                "grid": "native dense public_time samples within PA support",
                "interpolation": "clean natural cubic amplitude and lifted positive phase",
                "public_complex_strain_interpolated": False,
                "positive_phase_knots": "2*pa_state[:,1] - unwrap(angle(mode*exp(2j*pa_state[:,1])))",
            },
            "fixture": {"path": str(FIXTURE), "sha256": fixture_sha},
            "source_hashes_before": before,
            "source_hashes_after": after,
            "source_hashes_stable": before == after,
            "citation": "Damour, Nagar & Bernuzzi, arXiv:1212.4357v2, Eq. (23)",
            "tail_reference_radius": "r0=2GM/sqrt(e)",
            "correction": "mode_after = mode_before * exp(-i*m*Hreal*Omega)",
            "mode_expected_max_abs_error": float(torch.max(torch.abs(mode - expected))),
            "mode_amplitude_max_abs_error": float(torch.max(torch.abs(mode.abs() - mode_before.abs()))),
            "amplitude_api_max_abs_error": float(torch.max(torch.abs(amplitudes[(2, 2)] - mode.abs()))),
            "infinity_flux_identity_max_abs_error": float(torch.max(torch.abs(infinity_flux - amplitude_flux))),
            "gauge_unit_modulus_max_abs_error": float(torch.max(torch.abs(gauge.abs() - 1.0))),
            "public_projection_scale": scale,
            "public_phase_residual_first_rad": float(phase_residual[0]),
            "public_phase_residual_last_rad": float(phase_residual[-1]),
            "public_phase_residual_max_abs_rad": float(np.max(np.abs(phase_residual))),
            "public_projection_relative_l2": float(relative_l2),
            "public_amplitude_relative_l2": float(amplitude_relative_l2),
            "public_amplitude_relative_max_abs": float(amplitude_relative_max_abs),
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    log_path = report_path.with_suffix(".log")
    log_path.write_text(
        f"pid={result['pid']} command={result['command']}\n"
        f"mode_expected_max_abs_error={result['mode_expected_max_abs_error']}\n"
        f"public_phase_residual_max_abs_rad={result['public_phase_residual_max_abs_rad']}\n"
        f"public_amplitude_relative_l2={result['public_amplitude_relative_l2']}\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = _run(args.report)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
