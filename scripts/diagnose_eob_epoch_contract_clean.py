"""Measure public EOB epochs and clean attachment events on paired grids.

This is a read-only, clean-room diagnostic.  The trajectory, post-peak state,
and public ``(2, 2)`` mode are supplied by the public LALSimulation call and
loaded as copies.  The clean Hamiltonian, flux, waveform, event, NQC, remnant,
and attachment helpers are evaluated afresh for every spacing.  No source
waveform is modified and no time or phase offset is fitted.

The public mode has an epoch in seconds while the auxiliary trajectory starts
at the initial time ``t/M = 0``.  Attachment therefore receives the relative
output grid ``arange(n) * dt / Msec``.  Epoch coordinates are retained as a
separate coordinate for reporting and for the public amplitude maximum.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from torchwave.constants import C_SI, MPC_SI, MSUN_SEC
from torchwave.approximants.eob.attachment import attach_seobnrv4_22
from torchwave.approximants.eob.nqc import seobnrv4_nqc_targets
from torchwave.approximants.eob.published_waveform import (
    published_mode22_and_frequency,
)
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.eob.remnant import (
    eob_final_mass_fraction,
    hofmann_final_spin,
)
from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm


REPO_ROOT = Path(__file__).resolve().parents[1]
MTSUN = MSUN_SEC
TOTAL_MASS = 50.0
MSEC = TOTAL_MASS * MTSUN
DISTANCE_MPC = 100.0
MODE_SCALE = MSEC * C_SI / (DISTANCE_MPC * MPC_SI)
OUT = REPO_ROOT / (
    "docs/reviews/2026-09-13-parity-review/verification/eob/"
    "eob-epoch-contract-clean.json"
)
LABELS = ("dt_0p25M", "dt_0p5M", "dt_1M", "dt_1_4096s")
DATA_DIR = Path("/tmp")

# These are source files used by this diagnostic or by the clean public
# attachment path.  Hashing the wrapper records how the public black boxes
# were generated; hashing the direct physics dependencies makes the report
# reproducible when a clean-room helper changes.
DEPENDENCIES = (
    "scripts/diagnose_eob_epoch_contract_clean.py",
    "scripts/diagnose_eob_clean_hamiltonian_public.py",
    "src/torchwave/constants.py",
    "src/torchwave/approximants/eob/attachment.py",
    "src/torchwave/approximants/eob/events.py",
    "src/torchwave/approximants/eob/nqc.py",
    "src/torchwave/approximants/eob/published_waveform.py",
    "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
    "src/torchwave/approximants/eob/seobnrv4_flux.py",
    "src/torchwave/approximants/eob/seobnrv4_mode22.py",
    "src/torchwave/approximants/eob/factorized_modes.py",
    "src/torchwave/approximants/eob/merger_ringdown.py",
    "src/torchwave/approximants/eob/remnant.py",
    "src/torchwave/approximants/kerr_qnm_generated.py",
)


def sha(path: Path | str) -> str:
    """Return the SHA256 of a file without reading it through NumPy/Torch."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_number(value):
    """Convert a scalar NumPy/Torch value to a JSON-native number."""

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    if isinstance(value, np.generic):
        value = value.item()
    return float(value)


def quadratic_peak(x, y, i):
    """Return a three-point quadratic peak and fractional sample offset.

    This is deliberately retained as a diagnostic stencil.  It is not used
    by ``attach_seobnrv4_22``, whose event is the clean degree-six event.
    """

    if i <= 0 or i + 1 >= len(y):
        return float(x[i]), 0.0
    h = float(x[i + 1] - x[i])
    den = float(y[i - 1] - 2 * y[i] + y[i + 1])
    off = 0.5 * float(y[i - 1] - y[i + 1]) / den if den else 0.0
    return float(x[i] + off * h), float(off)


def local_polynomial_peak(x, y, i, degree):
    """Refine a sample maximum with a degree-4/6 local polynomial.

    The returned value is a convergence diagnostic only.  The attachment
    event remains the independently evaluated clean ``events.py`` degree-six
    event, including its derivative bracket and maximum checks.
    """

    if degree not in (4, 6):
        raise ValueError("diagnostic peak degree must be 4 or 6")
    count = degree + 1
    if len(y) < count:
        return {
            "resolved": False,
            "degree": degree,
            "sample_index": int(i),
            "time": float(x[i]),
            "fractional_bin_offset": 0.0,
            "frequency_or_amplitude": float(y[i]),
            "candidate_count": 0,
        }
    start = max(0, min(int(i) - degree // 2, len(y) - count))
    stop = start + count
    nodes = np.asarray(x[start:stop], dtype=np.float64)
    ordinates = np.asarray(y[start:stop], dtype=np.float64)
    scale = float(nodes[-1] - nodes[0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("peak stencil must have strictly increasing times")
    origin = float(x[i])
    local = (nodes - origin) / scale
    coefficients = np.polynomial.polynomial.polyfit(local, ordinates, degree)
    derivative = np.polynomial.polynomial.polyder(coefficients)
    second = np.polynomial.polynomial.polyder(derivative)
    candidates = []
    for root in np.polynomial.polynomial.polyroots(derivative):
        if abs(float(root.imag)) > 1e-8:
            continue
        z = float(root.real)
        if local[0] - 1e-12 <= z <= local[-1] + 1e-12:
            curvature = float(np.polynomial.polynomial.polyval(z, second))
            if curvature < 0:
                value = float(np.polynomial.polynomial.polyval(z, coefficients))
                candidates.append((value, z, curvature))
    if not candidates:
        return {
            "resolved": False,
            "degree": degree,
            "sample_index": int(i),
            "time": origin,
            "fractional_bin_offset": 0.0,
            "frequency_or_amplitude": float(y[i]),
            "candidate_count": 0,
            "stencil_start_index": int(start),
            "stencil_end_index": int(stop - 1),
        }
    value, z, curvature = max(candidates, key=lambda item: item[0])
    time = origin + z * scale
    return {
        "resolved": True,
        "degree": degree,
        "sample_index": int(i),
        "time": float(time),
        "fractional_bin_offset": float((time - origin) / (x[i + 1] - x[i]))
        if i + 1 < len(x)
        else 0.0,
        "frequency_or_amplitude": value,
        "curvature_in_local_coordinate": curvature,
        "candidate_count": len(candidates),
        "stencil_start_index": int(start),
        "stencil_end_index": int(stop - 1),
    }


def _peak_record(x, y, *, label):
    """Return sample, quadratic, and optional polynomial peak records."""

    i = int(np.argmax(y))
    quadratic_time, quadratic_offset = quadratic_peak(x, y, i)
    record = {
        "sample_index": i,
        "sample_time": float(x[i]),
        "sample_value": float(y[i]),
        "quadratic": {
            "degree": 2,
            "time": quadratic_time,
            "fractional_bin_offset": quadratic_offset,
            "value_at_sample": float(y[i]),
        },
        "stencil_diagnostics": {},
        "coordinate_label": label,
    }
    for stencil_degree in (4, 6):
        record["stencil_diagnostics"][f"degree{stencil_degree}"] = (
            local_polynomial_peak(x, y, i, stencil_degree)
        )
    return record


def _event_record(attached, tin, omega):
    """Name the attachment's degree-six event and diagnostic alternatives."""

    oi = int(attached.peak.sample_index.detach().cpu().item())
    om = np.asarray(omega.detach().cpu().numpy(), dtype=np.float64)
    quadratic_time, quadratic_offset = quadratic_peak(tin, om, oi)
    return {
        "degree6_attachment_event": {
            "time_M_initial_origin": _json_number(attached.peak.time),
            "frequency": _json_number(attached.peak.frequency),
            "curvature": _json_number(attached.peak.curvature),
            "sample_index": oi,
            "sample_time_M_initial_origin": float(tin[oi]),
            "fractional_bin_offset_not_available_from_attachment_api": None,
        },
        "quadratic_peak_diagnostic": {
            "time_M_initial_origin": quadratic_time,
            "fractional_bin_offset": quadratic_offset,
            "sample_index": oi,
            "frequency_at_sample": float(om[oi]),
        },
        "stencil_diagnostics": {
            f"degree{degree}": local_polynomial_peak(tin, om, oi, degree)
            for degree in (4, 6)
        },
    }


def _coefficients(attached):
    coeff = attached.calibration.coefficients
    amplitude = np.asarray(coeff.amplitude.detach().cpu().numpy(), dtype=np.float64)
    phase = np.asarray(coeff.phase.detach().cpu().numpy(), dtype=np.float64)
    return {
        "amplitude": amplitude.tolist(),
        "phase": phase.tolist(),
        "vector_order": ["a1", "a2", "a3", "b1", "b2"],
        "vector": np.concatenate((amplitude, phase)).tolist(),
        "sampling_condition": _json_number(attached.calibration.sampling_condition),
        "amplitude_condition": _json_number(attached.calibration.amplitude_condition),
        "phase_condition": _json_number(attached.calibration.phase_condition),
        "amplitude_residual_l2": float(
            torch.linalg.vector_norm(attached.calibration.amplitude_residual)
            .detach()
            .cpu()
        ),
        "phase_residual_l2": float(
            torch.linalg.vector_norm(attached.calibration.phase_residual)
            .detach()
            .cpu()
        ),
    }


def _postmatch_residual(
    clean_mode, public_mode, output_time_M, match_time_M, *, epoch_M=None
):
    """Measure the same-index, same-grid complex residual after matching.

    ``clean_mode`` has the public 100 Mpc SI scale applied.  Since the public
    epoch was removed only when constructing the relative attachment grid,
    identical indices have identical physical sample times.  No interpolation,
    time shift, phase shift, or fit is applied here.
    """

    clean_mode = np.asarray(clean_mode, dtype=np.complex128)
    public_mode = np.asarray(public_mode, dtype=np.complex128)
    output_time_M = np.asarray(output_time_M, dtype=np.float64)
    if clean_mode.shape != public_mode.shape or clean_mode.shape != output_time_M.shape:
        raise ValueError("clean mode, public mode, and output grid must have equal shape")
    mask = output_time_M >= float(match_time_M)
    if not bool(mask.any()):
        return {"available": False, "reason": "output grid ends before NQC match"}
    clean = clean_mode[mask]
    public = public_mode[mask]
    residual = clean - public
    nonzero = np.abs(public) > np.finfo(np.float64).tiny
    phase = np.angle(clean[nonzero] * np.conj(public[nonzero]))
    public_norm = float(np.linalg.norm(public))
    peak = float(np.max(np.abs(public))) if public.size else 0.0
    epoch_offset = float(epoch_M) if epoch_M is not None else 0.0
    return {
        "available": True,
        "comparison": "same output indices after the relative-grid epoch mapping",
        "time_shift_applied_M": 0.0,
        "phase_shift_applied_rad": 0.0,
        "first_sample_index": int(np.flatnonzero(mask)[0]),
        "sample_count": int(mask.sum()),
        "support_M_initial_origin": [
            float(output_time_M[mask][0]),
            float(output_time_M[mask][-1]),
        ],
        "public_support_M_epoch_coordinate": [
            float(epoch_offset + output_time_M[mask][0]),
            float(epoch_offset + output_time_M[mask][-1]),
        ],
        "raw_complex_l2": float(np.linalg.norm(residual)),
        "raw_relative_l2": float(np.linalg.norm(residual) / public_norm)
        if public_norm
        else None,
        "max_abs": float(np.max(np.abs(residual))) if residual.size else None,
        "max_abs_over_public_peak": float(np.max(np.abs(residual)) / peak)
        if peak
        else None,
        "phase_rms_rad": float(np.sqrt(np.mean(phase**2)))
        if phase.size
        else None,
        "phase_max_abs_rad": float(np.max(np.abs(phase))) if phase.size else None,
        "amplitude_ratio_mean": float(
            np.mean(np.abs(clean[nonzero]) / np.abs(public[nonzero]))
        )
        if nonzero.any()
        else None,
    }


def _library_versions():
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    for module_name, distribution_name in (
        ("lal", "lalsuite"),
        ("lalsimulation", "lalsuite"),
    ):
        try:
            module = __import__(module_name)
        except Exception:
            versions[module_name] = "unavailable"
            continue
        value = getattr(module, "__version__", None)
        if value is None:
            try:
                value = metadata.version(distribution_name)
            except metadata.PackageNotFoundError:
                value = "unknown"
        versions[module_name] = str(value)
    return versions


def _setup_clean_physics():
    dtype = torch.float64
    z = torch.tensor(0.0, dtype=dtype)
    nu = torch.tensor(0.24, dtype=dtype)
    ham = SEOBNRv4Hamiltonian(
        nu, z, z, delta=torch.tensor(0.2, dtype=dtype)
    )
    flux = SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8")
    mf = eob_final_mass_fraction(
        torch.tensor(0.6, dtype=dtype),
        torch.tensor(0.4, dtype=dtype),
        z,
        z,
    )
    af = hofmann_final_spin(
        torch.tensor(0.6, dtype=dtype),
        torch.tensor(0.4, dtype=dtype),
        z,
        z,
        fit_order=(3, 4),
    )
    wr, wd = dimensionless_qnm(2, 2, 0, af)
    sigma = torch.complex(wr / mf, -wd / mf)
    return {
        "dtype": dtype,
        "z": z,
        "nu": nu,
        "ham": ham,
        "flux": flux,
        "mf": mf,
        "af": af,
        "sigma": sigma,
        "targets": seobnrv4_nqc_targets(nu, z),
    }


def _load_public_dataset(label):
    path = DATA_DIR / f"torchwave-public-eob-{label}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; generate it through public_call in "
            "diagnose_eob_clean_hamiltonian_public.py"
        )
    with np.load(path) as data:
        # Every array is copied so subsequent Torch calls cannot mutate the
        # loaded NPZ-backed memory, even if a future helper changes in-place.
        state = np.asarray(data["state"], dtype=np.float64).copy()
        postpeak = np.asarray(data["postpeak_state"], dtype=np.float64).copy()
        mode = np.asarray(data["mode22"], dtype=np.complex128).copy()
        epoch_s = float(np.asarray(data["epoch_s"]).item())
        dt_s = float(np.asarray(data["dt_s"]).item())
    if state.ndim != 2 or state.shape[0] != 5:
        raise ValueError(f"{label}: expected state shape (5,N), got {state.shape}")
    if postpeak.ndim != 2 or postpeak.shape[0] != 5:
        raise ValueError(
            f"{label}: expected postpeak_state shape (5,N), got {postpeak.shape}"
        )
    if mode.ndim != 1 or len(mode) == 0:
        raise ValueError(f"{label}: expected one-dimensional mode array")
    if not np.isfinite(state).all() or not np.isfinite(postpeak).all():
        raise ValueError(f"{label}: public state contains non-finite values")
    if not np.isfinite(mode).all():
        raise ValueError(f"{label}: public mode contains non-finite values")
    if not np.all(np.diff(state[0]) > 0) or not np.all(np.diff(postpeak[0]) > 0):
        raise ValueError(f"{label}: public state times are not strictly increasing")
    if not np.isfinite(epoch_s) or not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError(f"{label}: invalid public epoch/dt metadata")
    return {
        "label": label,
        "path": path,
        "sha256": sha(path),
        "state": state,
        "postpeak_state": postpeak,
        "mode": mode,
        "epoch_s": epoch_s,
        "dt_s": dt_s,
    }


def _evaluate_dataset(data, physics):
    dtype = physics["dtype"]
    nu = physics["nu"]
    z = physics["z"]
    ham = physics["ham"]
    flux = physics["flux"]
    sigma = physics["sigma"]
    post = data["postpeak_state"]
    tin = post[0].copy()
    state = torch.from_numpy(post[1:].T.copy()).to(dtype)
    time = torch.from_numpy(tin.copy()).to(dtype)
    omega = flux.from_hamiltonian(state, ham)["omega"]
    base, frequency = published_mode22_and_frequency(
        state,
        ham,
        flux,
        spin_v6="none",
        include_pan_v8=True,
        include_positive_chi_540=True,
    )
    dt_M = data["dt_s"] / MSEC
    # This is the coordinate fix under test.  The public epoch is retained
    # only as metadata/physical reporting coordinate, not passed to attach.
    output_time_M = np.arange(data["mode"].size, dtype=np.float64) * dt_M
    attached = attach_seobnrv4_22(
        time,
        state,
        omega,
        base,
        frequency,
        nu,
        z,
        sigma,
        torch.from_numpy(output_time_M.copy()).to(dtype),
    )
    event = _event_record(attached, tin, omega)
    event_time_M = float(attached.peak.time.detach().cpu().item())
    match_time_M = float(attached.match_time.detach().cpu().item())
    epoch_M = data["epoch_s"] / MSEC
    mode_amplitude = np.abs(data["mode"])
    amp_peak = _peak_record(
        output_time_M,
        mode_amplitude,
        label="relative output time from public mode sample zero",
    )
    physical_amp = json.loads(json.dumps(amp_peak))
    physical_amp["sample_time"] = float(epoch_M + amp_peak["sample_time"])
    physical_amp["quadratic"]["time"] = float(
        epoch_M + amp_peak["quadratic"]["time"]
    )
    for value in physical_amp["stencil_diagnostics"].values():
        if value.get("resolved"):
            value["time"] = float(epoch_M + value["time"])
    physical_amp["coordinate_label"] = (
        "public mode epoch coordinate: epoch_s + relative output time"
    )
    coefficients = _coefficients(attached)
    clean_mode = (
        attached.mode.detach().cpu().numpy().astype(np.complex128) * MODE_SCALE
    )
    residual = _postmatch_residual(
        clean_mode,
        data["mode"],
        output_time_M,
        match_time_M,
        epoch_M=epoch_M,
    )
    event["degree6_attachment_event"]["time_M_public_epoch_coordinate"] = float(
        epoch_M + event_time_M
    )
    event["degree6_attachment_event"]["sample_time_M_public_epoch_coordinate"] = float(
        epoch_M + tin[event["degree6_attachment_event"]["sample_index"]]
    )
    event["quadratic_peak_diagnostic"]["time_M_public_epoch_coordinate"] = float(
        epoch_M + event["quadratic_peak_diagnostic"]["time_M_initial_origin"]
    )
    for value in event["stencil_diagnostics"].values():
        if value.get("resolved"):
            value["time_M_public_epoch_coordinate"] = float(epoch_M + value["time"])
    return {
        "npz_path": str(data["path"]),
        "npz_sha256": data["sha256"],
        "npz_size_bytes": int(data["path"].stat().st_size),
        "raw_arrays": {
            "keys": ["state", "postpeak_state", "mode22", "epoch_s", "dt_s"],
            "state_shape": list(data["state"].shape),
            "postpeak_state_shape": list(data["postpeak_state"].shape),
            "mode22_shape": list(data["mode"].shape),
            "loaded_arrays_copied": True,
        },
        "dt_s": data["dt_s"],
        "dt_M": dt_M,
        "epoch_s": data["epoch_s"],
        "epoch_M": epoch_M,
        "relative_output_grid": {
            "passed_to_attachment": True,
            "definition": "arange(output_length) * dt_s / Msec",
            "first_last_M": [float(output_time_M[0]), float(output_time_M[-1])],
            "first_last_s": [
                float(output_time_M[0] * MSEC),
                float(output_time_M[-1] * MSEC),
            ],
        },
        "public_epoch_output_grid": {
            "definition": "epoch_s + arange(output_length) * dt_s",
            "first_last_s": [
                float(data["epoch_s"]),
                float(data["epoch_s"] + output_time_M[-1] * MSEC),
            ],
            "first_last_M": [
                float(epoch_M),
                float(epoch_M + output_time_M[-1]),
            ],
        },
        "trajectory_support_M_initial_origin": [
            float(data["state"][0, 0]),
            float(data["state"][0, -1]),
        ],
        "postpeak_support_M_initial_origin": [float(tin[0]), float(tin[-1])],
        "degree6_event_and_peak_diagnostics": event,
        "nqc_match": {
            "time_M_initial_origin": match_time_M,
            "time_M_public_epoch_coordinate": float(epoch_M + match_time_M),
            "minus_degree6_event_M": float(match_time_M - event_time_M),
            "minus_quadratic_peak_M": float(
                match_time_M
                - event["quadratic_peak_diagnostic"]["time_M_initial_origin"]
            ),
        },
        "public_mode_amplitude_peak": {
            "relative_output_coordinate": amp_peak,
            "public_epoch_coordinate": physical_amp,
        },
        "nqc_coefficients": coefficients,
        "postmatch_unshifted_complex_residual": residual,
        "clean_attachment": {
            "mode_scale_100Mpc": MODE_SCALE,
            "mode_length": int(attached.mode.shape[-1]),
            "match_phase_rad": _json_number(attached.match_phase),
        },
    }


def _sampling_dependence(results):
    fields = {
        "public_epoch_M": lambda row: row["epoch_M"],
        "degree6_event_time_M_initial_origin": lambda row: row[
            "degree6_event_and_peak_diagnostics"
        ]["degree6_attachment_event"]["time_M_initial_origin"],
        "degree6_event_time_M_public_epoch_coordinate": lambda row: row[
            "degree6_event_and_peak_diagnostics"
        ]["degree6_attachment_event"]["time_M_public_epoch_coordinate"],
        "quadratic_event_time_M_initial_origin": lambda row: row[
            "degree6_event_and_peak_diagnostics"
        ]["quadratic_peak_diagnostic"]["time_M_initial_origin"],
        "public_amplitude_quadratic_time_M_epoch_coordinate": lambda row: row[
            "public_mode_amplitude_peak"
        ]["public_epoch_coordinate"]["quadratic"]["time"],
        "nqc_match_time_M_initial_origin": lambda row: row["nqc_match"][
            "time_M_initial_origin"
        ],
        "postmatch_residual_raw_relative_l2": lambda row: row[
            "postmatch_unshifted_complex_residual"
        ]["raw_relative_l2"],
    }
    coefficient_fields = {
        "nqc_amplitude": lambda row: np.asarray(
            row["nqc_coefficients"]["amplitude"], dtype=np.float64
        ),
        "nqc_phase": lambda row: np.asarray(
            row["nqc_coefficients"]["phase"], dtype=np.float64
        ),
    }
    out = {"pairwise": {}, "ranges": {}}
    for field, getter in fields.items():
        values = [getter(results[label]) for label in LABELS]
        finite = [float(value) for value in values if value is not None]
        out["ranges"][field] = {
            "minimum": min(finite) if finite else None,
            "maximum": max(finite) if finite else None,
            "max_minus_min": (max(finite) - min(finite)) if finite else None,
        }
    for field, getter in coefficient_fields.items():
        values = {label: getter(results[label]) for label in LABELS}
        stack = np.stack(tuple(values.values()))
        out["ranges"][field] = {
            "component_max": np.max(stack, axis=0).tolist(),
            "component_min": np.min(stack, axis=0).tolist(),
            "component_max_minus_min": (
                np.max(stack, axis=0) - np.min(stack, axis=0)
            ).tolist(),
            "component_max_abs": np.max(np.abs(stack), axis=0).tolist(),
            "vector_max_abs": float(np.max(np.abs(stack))),
            "vector_range_l2": float(
                np.linalg.norm(np.max(stack, axis=0) - np.min(stack, axis=0))
            ),
        }
    for left, right in itertools.combinations(LABELS, 2):
        key = f"{left}_minus_{right}"
        row = {}
        for field, getter in fields.items():
            lv, rv = getter(results[left]), getter(results[right])
            row[field] = (
                float(lv - rv) if lv is not None and rv is not None else None
            )
        for field, getter in coefficient_fields.items():
            delta = getter(results[left]) - getter(results[right])
            row[field] = {
                "component_delta": delta.tolist(),
                "max_abs": float(np.max(np.abs(delta))),
                "l2": float(np.linalg.norm(delta)),
            }
        out["pairwise"][key] = row
    return out


def main():
    physics = _setup_clean_physics()
    results = {}
    for label in LABELS:
        results[label] = _evaluate_dataset(_load_public_dataset(label), physics)
    report = {
        "provenance": (
            "Public lalsimulation SimIMRSpinAlignedEOBModes output was consumed "
            "as black-box state/postpeak/mode/epoch data for 30+20 Msun, "
            "nonspinning, fstart=20 Hz, distance=100 Mpc. For each spacing, "
            "the clean SEOBNRv4 Hamiltonian/flux/waveform/event/NQC/remnant/QNM "
            "helpers were evaluated afresh. Attachment receives the initial-time "
            "relative grid; public epoch coordinates remain separate. The "
            "postmatch residual is same-index complex subtraction with zero time "
            "and phase shifts, and all quadratic/degree4/degree6 alternatives are "
            "numerical convergence diagnostics only."
        ),
        "primary_reference": (
            "Bohé et al., Improved effective-one-body model for coalescing "
            "nonprecessing binary black holes, arXiv:1611.03703 (equation context)."
        ),
        "public_parameters": {
            "approximant_api": "lalsimulation.SimIMRSpinAlignedEOBModes",
            "m1_msun": 30.0,
            "m2_msun": 20.0,
            "chi1": 0.0,
            "chi2": 0.0,
            "fstart_hz": 20.0,
            "distance_mpc": DISTANCE_MPC,
            "mode": [2, 2],
            "approximant_version": 4,
            "inclination_rad": 0.0,
            "phase_rad": 0.0,
            "eccentricity": 0.0,
            "tidal_lambda1": 0.0,
            "tidal_lambda2": 0.0,
            "spacings": {
                "dt_0p25M": 0.25,
                "dt_0p5M": 0.5,
                "dt_1M": 1.0,
                "dt_1_4096s": 1.0 / 4096.0 / MSEC,
            },
        },
        "clean_parameters": {
            "nu": 0.24,
            "delta": 0.2,
            "chi": 0.0,
            "spin_truncation": "generic-leading-ell8",
            "spin_v6": "none",
            "include_pan_v8": True,
            "include_positive_chi_540": True,
            "final_spin_fit_order": [3, 4],
            "mode_scale_100Mpc": MODE_SCALE,
            "qnm_mode": [2, 2, 0],
        },
        "mass_assumption": {
            "total_mass_msun": TOTAL_MASS,
            "Msec": MSEC,
            "MTSUN_s": MTSUN,
        },
        "coordinate_contract": {
            "input_time": "postpeak_state[0], initial total-mass coordinates starting near t/M=0",
            "attachment_output_time": "arange(len(mode22)) * dt_s / Msec",
            "public_epoch_time": "epoch_s + arange(len(mode22)) * dt_s",
            "residual_alignment": "same output index after removing only the public epoch from attachment coordinates",
        },
        "libraries": _library_versions(),
        "python_executable": sys.executable,
        "host": platform.node(),
        "dependency_sha256": {
            path: sha(REPO_ROOT / path) for path in DEPENDENCIES
        },
        "results": results,
        "sampling_dependence": _sampling_dependence(results),
        "limitations": [
            "The public epoch is reported as returned by LALSimulation; its physical interpretation is not inferred beyond the explicit grid mapping.",
            "The four black-box arrays are compared at identical output indices. No interpolation, fitted offset, or parity-based stencil selection is used.",
            "The diagnostic event is the clean degree-six event from events.py; quadratic and degree-4/6 polynomial peaks are reported only to show numerical stencil sensitivity.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
