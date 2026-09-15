"""Test observable sample-bracketing hypotheses for public EOB epochs.

The public epoch is compared with ``-epoch`` in the initial-origin total-mass
coordinate.  This diagnostic uses only the eight saved black-box public
outputs from ``diagnose_eob_public_epoch_sampling_clean.py``.  For each
returned postpeak grid, the clean Hamiltonian evaluates the orbital
frequency.  A fixed local polynomial derivative is then sampled at the
clean-frequency sample maximum and its immediately following high-resolution
sample.

The candidate convention is deliberately discrete: start with that adjacent
sample interval, apply the stated generic bisection update rule a
predeclared number of times (or until a predeclared width), and return the
midpoint.  Same-sign endpoints are reported as nonbracketing; their candidate
is not called a physical root.  No epoch or waveform shift is fitted or
installed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from diagnose_eob_public_phase_peak_clean import _clean_event, _setup_clean_physics
from torchwave.constants import MSUN_SEC


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/tmp")
OUT = REPO_ROOT / (
    "docs/reviews/2026-09-13-parity-review/verification/eob/"
    "eob-public-epoch-bisection-clean.json"
)
TOTAL_MASS_MSUN = 50.0
MSEC = TOTAL_MASS_MSUN * MSUN_SEC
GPS_QUANTUM_S = 1.0e-9
GPS_QUANTUM_M = GPS_QUANTUM_S / MSEC
FIXED_ITERATIONS = (8, 10, 12, 14)
TOLERANCE_M = 1.0e-5
MAX_TOLERANCE_ITERATIONS = 64

INPUTS = (
    ("dt_0p90M", DATA_DIR / "torchwave-public-eob-epoch-sweep-dt_0p90M.npz"),
    ("dt_0p95M", DATA_DIR / "torchwave-public-eob-epoch-sweep-dt_0p95M.npz"),
    ("dt_0p99M", DATA_DIR / "torchwave-public-eob-epoch-sweep-dt_0p99M.npz"),
    ("dt_1p00M", DATA_DIR / "torchwave-public-eob-epoch-sweep-dt_1p00M.npz"),
    ("dt_1p01M", DATA_DIR / "torchwave-public-eob-epoch-sweep-dt_1p01M.npz"),
    ("dt_1p05M", DATA_DIR / "torchwave-public-eob-epoch-sweep-dt_1p05M.npz"),
    ("dt_1p10M", DATA_DIR / "torchwave-public-eob-epoch-sweep-dt_1p10M.npz"),
    ("dt_1_4096s_exact", DATA_DIR / "torchwave-public-eob-epoch-sweep-dt_1_4096s_exact.npz"),
)

DEPENDENCIES = (
    "scripts/diagnose_eob_public_epoch_bisection_clean.py",
    "scripts/diagnose_eob_public_epoch_sampling_clean.py",
    "scripts/diagnose_eob_public_phase_peak_clean.py",
    "src/torchwave/constants.py",
    "src/torchwave/approximants/eob/events.py",
    "src/torchwave/approximants/eob/hamiltonian.py",
    "src/torchwave/approximants/eob/integrator.py",
    "src/torchwave/approximants/eob/radiation_reaction.py",
    "src/torchwave/approximants/eob/seobnrv4_flux.py",
    "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
)


def sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _library_versions() -> dict[str, str]:
    # Keep the pure observable-array helpers importable in environments where
    # the optional public LAL bindings are unavailable.  The diagnostic's
    # numerical path does not need these modules until this metadata is built.
    try:
        import lal
    except ImportError:
        lal = None
    try:
        import lalsimulation as ls
    except ImportError:
        ls = None
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "lal": getattr(lal, "__version__", "unknown"),
        "lalsimulation": getattr(ls, "__version__", "unknown"),
    }
    for module_name in ("lal", "lalsimulation"):
        if versions[module_name] != "unknown":
            continue
        try:
            versions[module_name] = str(metadata.version("lalsuite"))
        except metadata.PackageNotFoundError:
            versions[module_name] = "unknown"
    return versions


def _clean_omega(postpeak_state: np.ndarray, physics: dict) -> np.ndarray:
    """Evaluate clean omega on a private copy of one public state array."""

    state_t = torch.from_numpy(
        np.asarray(postpeak_state[1:].T, dtype=np.float64).copy()
    )
    with torch.inference_mode():
        omega = physics["flux"].from_hamiltonian(state_t, physics["ham"])["omega"]
    return omega.detach().cpu().numpy().astype(np.float64, copy=True)


def _local_polynomial_derivative(
    time: np.ndarray,
    ordinate: np.ndarray,
    anchor_index: int,
    x: float,
    degree: int,
) -> float:
    """Evaluate a fixed-stencil polynomial derivative at ``x``.

    Degree 6 uses the same seven-point local structure as the clean event
    detector.  Degree 3 uses the four contiguous points ``i-1:i+3`` around
    the sample maximum.  Coordinates are normalized by the stencil span so
    derivatives remain well-conditioned in the large initial-origin chart.
    """

    count = degree + 1
    if degree not in (3, 6):
        raise ValueError("derivative degree must be 3 or 6")
    if time.ndim != 1 or ordinate.ndim != 1 or time.size != ordinate.size:
        raise ValueError("time and ordinate must be one-dimensional and matched")
    if time.size < count:
        raise ValueError("derivative stencil is longer than the sample series")
    start = max(0, min(int(anchor_index) - degree // 2, time.size - count))
    stop = start + count
    nodes = np.asarray(time[start:stop], dtype=np.float64)
    values = np.asarray(ordinate[start:stop], dtype=np.float64)
    span = float(nodes[-1] - nodes[0])
    if not np.isfinite(span) or span <= 0:
        raise ValueError("derivative stencil needs increasing finite times")
    reference = float(time[anchor_index])
    local = (nodes - reference) / span
    coefficients = np.polynomial.polynomial.polyfit(local, values, degree)
    derivative = np.polynomial.polynomial.polyder(coefficients)
    return float(
        np.polynomial.polynomial.polyval((float(x) - reference) / span, derivative)
        / span
    )


def _endpoint_derivative_data(
    time: np.ndarray, omega: np.ndarray, sample_index: int, degree: int
) -> dict:
    """Return endpoint derivatives and the clean adjacent-sample interval."""

    if sample_index < 0 or sample_index + 1 >= time.size:
        raise ValueError("sample maximum needs an immediately following sample")
    left = float(time[sample_index])
    right = float(time[sample_index + 1])
    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        raise ValueError("sample maximum interval must be finite and increasing")
    f_left = _local_polynomial_derivative(time, omega, sample_index, left, degree)
    f_right = _local_polynomial_derivative(time, omega, sample_index, right, degree)
    product = f_left * f_right
    if product < 0:
        bracket_status = "valid_sign_bracket"
    elif product > 0:
        sign = "positive" if f_left > 0 else "negative" if f_left < 0 else "zero"
        bracket_status = f"nonbracketing_same_sign_{sign}"
    else:
        bracket_status = "endpoint_or_zero_product"
    return {
        "left_time_M_initial_origin": left,
        "right_time_M_initial_origin": right,
        "interval_width_M": right - left,
        "left_derivative_domega_dt_M2": f_left,
        "right_derivative_domega_dt_M2": f_right,
        "derivative_product": product,
        "bracket_status": bracket_status,
        "valid_sign_bracket": bool(product < 0),
    }


def _bisection_hypothesis(
    time: np.ndarray,
    omega: np.ndarray,
    sample_index: int,
    degree: int,
    *,
    iterations: int | None = None,
    tolerance_M: float | None = None,
) -> dict:
    """Apply the predeclared generic endpoint-product bisection convention."""

    if (iterations is None) == (tolerance_M is None):
        raise ValueError("choose exactly one of iterations or tolerance_M")
    if iterations is not None and iterations < 0:
        raise ValueError("iterations must be nonnegative")
    if tolerance_M is not None and (not np.isfinite(tolerance_M) or tolerance_M <= 0):
        raise ValueError("tolerance_M must be positive and finite")
    endpoint = _endpoint_derivative_data(time, omega, sample_index, degree)
    left = endpoint["left_time_M_initial_origin"]
    right = endpoint["right_time_M_initial_origin"]
    f_left = endpoint["left_derivative_domega_dt_M2"]
    f_right = endpoint["right_derivative_domega_dt_M2"]
    requested_iterations = iterations
    stopped_by_tolerance = False
    exact_zero = False
    updates = 0
    max_updates = iterations if iterations is not None else MAX_TOLERANCE_ITERATIONS
    for _ in range(max_updates):
        if tolerance_M is not None and right - left <= tolerance_M:
            stopped_by_tolerance = True
            break
        midpoint = (left + right) / 2.0
        f_midpoint = _local_polynomial_derivative(
            time, omega, sample_index, midpoint, degree
        )
        product = f_left * f_midpoint
        updates += 1
        if product > 0:
            # This is deliberately the generic endpoint-product rule.  With
            # same-sign endpoints it is a nonbracketing endpoint contraction,
            # not a claim that a root was found.
            left = midpoint
            f_left = f_midpoint
        elif product < 0:
            right = midpoint
            f_right = f_midpoint
        else:
            # A zero derivative is a valid observed endpoint root.  Collapse
            # the interval so the reported midpoint is that sampled root.
            left = midpoint
            right = midpoint
            f_left = f_midpoint
            f_right = f_midpoint
            exact_zero = True
            break
        if tolerance_M is not None and right - left <= tolerance_M:
            stopped_by_tolerance = True
            break
    candidate = (left + right) / 2.0
    return {
        "degree": degree,
        "derivative_method": f"fixed_local_degree_{degree}_polynomial",
        "requested_iterations": requested_iterations,
        "requested_tolerance_M": tolerance_M,
        "updates_performed": updates,
        "stopped_by_tolerance": stopped_by_tolerance,
        "exact_zero_encountered": exact_zero,
        "candidate_time_M_initial_origin": float(candidate),
        "final_interval_M_initial_origin": [float(left), float(right)],
        "final_interval_width_M": float(right - left),
        "final_left_derivative_domega_dt_M2": float(f_left),
        "final_right_derivative_domega_dt_M2": float(f_right),
        "initial_endpoint": endpoint,
    }


def _gps_quantization(epoch_s: float) -> dict:
    """Describe the nearest-nanosecond interval around the observed epoch."""

    epoch_ns_float = float(epoch_s / GPS_QUANTUM_S)
    nearest_ns = int(np.rint(epoch_ns_float))
    observed_residual_ns = epoch_ns_float - nearest_ns
    epoch_interval_s = [
        float((nearest_ns - 0.5) * GPS_QUANTUM_S),
        float((nearest_ns + 0.5) * GPS_QUANTUM_S),
    ]
    minus_epoch_M = -float(epoch_s) / MSEC
    minus_epoch_interval_M = [
        float(minus_epoch_M - 0.5 * GPS_QUANTUM_M),
        float(minus_epoch_M + 0.5 * GPS_QUANTUM_M),
    ]
    return {
        "quantum_s": GPS_QUANTUM_S,
        "quantum_M": GPS_QUANTUM_M,
        "half_width_s": 0.5 * GPS_QUANTUM_S,
        "half_width_M": 0.5 * GPS_QUANTUM_M,
        "observed_epoch_s": float(epoch_s),
        "nearest_integer_epoch_ns": nearest_ns,
        "observed_minus_nearest_epoch_ns": float(observed_residual_ns),
        "nearest_ns_interval_epoch_s": epoch_interval_s,
        "nearest_ns_interval_minus_epoch_coordinate_M": minus_epoch_interval_M,
        "interpretation": (
            "The interval is the nearest-GPS-nanosecond quantization interval "
            "used for a numerical comparison; it does not establish the "
            "producer's internal timestamp representation."
        ),
    }


def _load_input(label: str, path: Path) -> dict:
    """Load and copy one previously saved public output without aliasing it."""

    with np.load(path, allow_pickle=False) as loaded:
        required = {"postpeak_state", "epoch_s", "dt_s"}
        missing = required.difference(loaded.files)
        if missing:
            raise ValueError(f"{label}: missing input keys {sorted(missing)}")
        postpeak = np.asarray(loaded["postpeak_state"], dtype=np.float64).copy()
        epoch_s = float(np.asarray(loaded["epoch_s"]).item())
        dt_s = float(np.asarray(loaded["dt_s"]).item())
        requested_dt_M = (
            float(np.asarray(loaded["requested_dt_M"]).item())
            if "requested_dt_M" in loaded.files
            else float(dt_s / MSEC)
        )
    if postpeak.ndim != 2 or postpeak.shape[0] != 5 or postpeak.shape[1] < 9:
        raise ValueError(f"{label}: postpeak_state must have shape (5,N>=9)")
    if not np.isfinite(postpeak).all() or not np.isfinite(epoch_s) or dt_s <= 0:
        raise ValueError(f"{label}: nonfinite or invalid input")
    if not np.all(np.diff(postpeak[0]) > 0):
        raise ValueError(f"{label}: postpeak time is not strictly increasing")
    return {
        "postpeak_state": postpeak,
        "epoch_s": epoch_s,
        "dt_s": dt_s,
        "requested_dt_M": requested_dt_M,
    }


def _row(label: str, path: Path, data: dict, physics: dict) -> dict:
    postpeak = data["postpeak_state"]
    time = postpeak[0].copy()
    omega = _clean_omega(postpeak, physics)
    event = _clean_event(time, postpeak, physics)
    event_omega = np.asarray(event.pop("omega"), dtype=np.float64).copy()
    if not np.allclose(event_omega, omega, rtol=2e-13, atol=2e-15):
        raise ValueError(f"{label}: clean event and omega evaluations disagree")
    sample_index = int(np.argmax(omega))
    sample_time = float(time[sample_index])
    next_time = float(time[sample_index + 1])
    spacing = next_time - sample_time
    minus_epoch_M = -data["epoch_s"] / MSEC
    prediction = next_time - spacing / 2048.0
    gps = _gps_quantization(data["epoch_s"])
    hypotheses = {}
    for degree in (3, 6):
        degree_key = str(degree)
        hypotheses[degree_key] = {
            "fixed_iterations": {
                str(iterations): _bisection_hypothesis(
                    time, omega, sample_index, degree, iterations=iterations
                )
                for iterations in FIXED_ITERATIONS
            },
            "tolerance": {
                "1e-5M": _bisection_hypothesis(
                    time, omega, sample_index, degree, tolerance_M=TOLERANCE_M
                )
            },
        }
        for group in hypotheses[degree_key].values():
            for candidate in group.values():
                candidate_time = candidate["candidate_time_M_initial_origin"]
                candidate["candidate_minus_minus_epoch_M"] = float(
                    candidate_time - minus_epoch_M
                )
                candidate["candidate_minus_minus_epoch_ns"] = float(
                    (candidate_time - minus_epoch_M) * MSEC / GPS_QUANTUM_S
                )
                candidate["candidate_within_gps_half_interval"] = bool(
                    abs(candidate_time - minus_epoch_M) <= 0.5 * GPS_QUANTUM_M
                )
    return {
        "status": "ok",
        "input_path": str(path),
        "input_sha256": sha256(path),
        "requested_dt_M": float(data["requested_dt_M"]),
        "returned_dt_s": float(data["dt_s"]),
        "returned_dt_M": float(data["dt_s"] / MSEC),
        "postpeak_sample_count": int(time.size),
        "postpeak_spacing_M": float(np.median(np.diff(time))),
        "sample_maximum": {
            "index": sample_index,
            "time_M_initial_origin": sample_time,
            "omega": float(omega[sample_index]),
            "next_time_M_initial_origin": next_time,
            "next_minus_sample_M": spacing,
            "next_minus_dt_over_2048_prediction_M": prediction,
            "prediction_minus_minus_epoch_M": float(prediction - minus_epoch_M),
            "prediction_minus_minus_epoch_ns": float(
                (prediction - minus_epoch_M) * MSEC / GPS_QUANTUM_S
            ),
        },
        "clean_event": {
            key: value for key, value in event.items()
        },
        "clean_event_side_of_sample_max": (
            "before" if event["time_M_initial_origin"] < sample_time else
            "after" if event["time_M_initial_origin"] > sample_time else "equal"
        ),
        "public_epoch": {
            "epoch_s": float(data["epoch_s"]),
            "epoch_M": float(data["epoch_s"] / MSEC),
            "minus_epoch_M": float(minus_epoch_M),
            "minus_epoch_minus_sample_M": float(minus_epoch_M - sample_time),
            "minus_epoch_minus_clean_event_M": float(
                minus_epoch_M - event["time_M_initial_origin"]
            ),
            "gps_quantization": gps,
        },
        "bisection_hypotheses": hypotheses,
        "no_fitted_shift": True,
        "same_sign_candidates_are_not_roots": True,
    }


def main() -> None:
    physics = _setup_clean_physics()
    results = {}
    for label, path in INPUTS:
        try:
            data = _load_input(label, path)
            results[label] = _row(label, path, data, physics)
        except Exception as exc:
            # A missing/invalid saved row remains visible as a diagnostic
            # failure; the input design is not changed after inspection.
            results[label] = {
                "status": "failed",
                "input_path": str(path),
                "input_sha256": sha256(path) if path.exists() else None,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "no_fitted_shift": True,
            }

    dependency_sha256 = {
        path: sha256(REPO_ROOT / path) if (REPO_ROOT / path).exists() else None
        for path in DEPENDENCIES
    }
    successful = [row for row in results.values() if row.get("status") == "ok"]
    ten_degree6 = [
        row["bisection_hypotheses"]["6"]["fixed_iterations"]["10"]
        for row in successful
    ]
    ten_within_gps = sum(
        bool(row["candidate_within_gps_half_interval"])
        for row in ten_degree6
    )
    before_rows = [
        label for label, row in results.items()
        if row.get("status") == "ok" and row["clean_event_side_of_sample_max"] == "before"
    ]
    after_rows = [
        label for label, row in results.items()
        if row.get("status") == "ok" and row["clean_event_side_of_sample_max"] == "after"
    ]
    report = {
        "provenance": (
            "Bounded clean-room comparison of endpoint-product bisection hypotheses "
            "against eight previously saved public SEOBNRv4 output rows. The public "
            "API output is consumed only as copied arrays and epoch metadata; clean "
            "SEOBNRv4 Hamiltonian/flux code evaluates omega. No LAL source or private "
            "waveform implementation is inspected."
        ),
        "primary_reference": (
            "Bohé et al., Improved effective-one-body model for coalescing "
            "nonprecessing binary black holes, arXiv:1611.03703 (equation context)."
        ),
        "public_output_inputs": {
            "labels": [label for label, _ in INPUTS],
            "source_report": "docs/reviews/2026-09-13-parity-review/verification/eob/eob-public-epoch-sampling-clean.json",
            "source_report_sha256": sha256(
                REPO_ROOT / "docs/reviews/2026-09-13-parity-review/verification/eob/eob-public-epoch-sampling-clean.json"
            ),
            "all_inputs_external_to_repository": True,
        },
        "public_parameters": {
            "api": "lalsimulation.SimIMRSpinAlignedEOBModes (source rows already saved)",
            "approximant": "SEOBNRv4 version 4",
            "m1_msun": 30.0,
            "m2_msun": 20.0,
            "chi1": 0.0,
            "chi2": 0.0,
            "fstart_hz": 20.0,
            "distance_mpc": 100.0,
            "mode": [2, 2],
        },
        "clean_parameters": {
            "hamiltonian": "SEOBNRv4Hamiltonian",
            "flux": "SEOBNRv4PublishedFlux.from_hamiltonian",
            "event": "orbital_frequency_peak degree-six local event detector",
            "derivative_cross_checks": [3, 6],
            "coordinate": "initial-origin total-mass M",
        },
        "endpoint_semantics": {
            "left": "clean omega argmax sample on returned postpeak_state[0]",
            "right": "immediately following returned postpeak_state[0] sample",
            "source": "observable returned samples only; no public implementation source inspection",
            "derivative": "clean local polynomial derivative on fixed contiguous stencil",
        },
        "bisection_algorithm": {
            "interval": "[left_sample, right_next_sample]",
            "update_rule": (
                "At each midpoint, if f(left)*f(mid)>0 move left=mid; if "
                "f(left)*f(mid)<0 move right=mid; return the interval midpoint. "
                "A zero midpoint collapses the interval."
            ),
            "fixed_iterations": list(FIXED_ITERATIONS),
            "tolerance_M": TOLERANCE_M,
            "tolerance_max_iterations": MAX_TOLERANCE_ITERATIONS,
            "candidate_status": (
                "A sign-changing endpoint pair is a valid derivative bracket. "
                "Same-sign endpoint candidates are retained as nonbracketing "
                "algorithm outputs and are not labeled roots."
            ),
            "no_residual_or_offset_selection": True,
        },
        "gps_quantization": {
            "assumed_quantum_s": GPS_QUANTUM_S,
            "assumed_quantum_M": GPS_QUANTUM_M,
            "interval_definition": "observed epoch +/- 0.5 ns, converted to -epoch initial-origin M coordinate",
            "not_exact_claim": True,
        },
        "libraries": _library_versions(),
        "python_executable": sys.executable,
        "host": platform.node(),
        "dependency_sha256": dependency_sha256,
        "results": results,
        "summary": {
            "successful_rows": len(successful),
            "failed_rows": len(results) - len(successful),
            "failed_labels": [label for label, row in results.items() if row.get("status") != "ok"],
            "event_before_sample_rows": before_rows,
            "event_after_sample_rows": after_rows,
            "degree6_fixed10_candidates_within_gps_half_interval": int(ten_within_gps),
            "degree6_fixed10_candidate_count": len(ten_degree6),
        },
        "interpretation": (
            "The next-sample-minus-dt/2048 value is the exact midpoint returned "
            "by ten generic updates when both endpoint derivatives have the same "
            "sign and the update keeps moving the left endpoint rightward. This "
            "is an implementation-convention hypothesis, not an inferred physical "
            "time shift. Sign-changing rows are evaluated as ordinary bisection "
            "brackets; same-sign rows retain their nonbracketing status."
        ),
        "runtime_recommendation": (
            "Do not install an epoch correction from these candidate comparisons. "
            "If compatibility with a public epoch becomes a product requirement, "
            "supporting an explicit, separately named discrete epoch convention "
            "could be considered only after an independent black-box validation; "
            "the continuous orbital-frequency event remains the clean runtime origin."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
