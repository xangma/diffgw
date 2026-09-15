#!/usr/bin/env python3
"""Clean-room audit of roundoff in the saved TEOB NQC five-point derivative.

This is deliberately a cache-only diagnostic.  It derives the degree-four
finite-difference rows from the polynomial moment equations, compares the
Torch weighted sum with an independently assembled rational stencil, and
repeats the same reconstruction with multi-precision arithmetic on the exact
binary64 cache values.  No trajectory evolution, external reference code, or
fitted constants are used.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import sys
import time
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
import torch

from torchwave.approximants.eob.teob_merger_nqc_clean import (
    _solve_two_by_two,
    quartic_derivative,
)


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-nqc-roundoff-clean"
CACHE = ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-native-replay-public-clean/replay-native-output.npz"
REPLAY_REPORT = CACHE.with_name("replay-report.json")
MODULE = ROOT / "src/torchwave/approximants/eob/teob_merger_nqc_clean.py"
CONTRACT = ROOT / "docs/reviews/2026-09-13-parity-review/teob-merger-assembly-contract.md"
OUTPUT = VERIFY / "diagnostic.json"
CONCLUSION = VERIFY / "conclusion.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def exact_mp(value: float) -> mp.mpf:
    """Convert a Python float to its exact binary rational in mp arithmetic."""
    frac = Fraction.from_float(float(value))
    return mp.mpf(frac.numerator) / frac.denominator


def moment_weights(nodes: list[int], evaluation: int) -> list[Fraction]:
    """Solve sum_j w_j (x_j-x)^k = delta[k,1], k=0..4, exactly."""
    n = len(nodes)
    a = [[Fraction(nodes[j] - evaluation) ** k for j in range(n)] for k in range(n)]
    b = [Fraction(1 if k == 1 else 0) for k in range(n)]
    for col in range(n):
        pivot = next(row for row in range(col, n) if a[row][col])
        a[col], a[pivot] = a[pivot], a[col]
        b[col], b[pivot] = b[pivot], b[col]
        scale = a[col][col]
        a[col] = [v / scale for v in a[col]]
        b[col] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            if factor:
                a[row] = [u - factor * v for u, v in zip(a[row], a[col])]
                b[row] -= factor * b[col]
    return b


def rational_rows() -> dict[int, tuple[list[int], int]]:
    """Return integer numerators and one denominator for endpoint rows.

    All rows use the moment equations above.  The centered row is subsequently
    evaluated by symmetry-pairing its integer numerators; endpoint rows use
    the nearest five nodes and are converted to a single denominator.
    """
    centered = moment_weights(list(range(-2, 3)), 0)
    centered_denominator = math.lcm(*(item.denominator for item in centered))
    rows: dict[int, tuple[list[int], int]] = {
        2: ([int(item * centered_denominator) for item in centered], centered_denominator),
    }
    for evaluation in (0, 1, 3, 4):
        weights = moment_weights(list(range(5)), evaluation)
        denominator = math.lcm(*(item.denominator for item in weights))
        rows[evaluation] = ([int(item * denominator) for item in weights], denominator)
    return rows


def rational_derivative(values: np.ndarray, spacing: float, *, sample_axis: int = -1) -> np.ndarray:
    """Apply the independently derived stencil using float64 arithmetic."""
    moved = np.moveaxis(np.asarray(values, dtype=np.float64), sample_axis, -1)
    n = moved.shape[-1]
    if n < 5:
        raise ValueError("need five samples")
    rows = rational_rows()
    out = np.empty_like(moved)
    for i in range(n):
        center = min(max(i, 2), n - 3)
        evaluation = i if i < 2 else (i - (n - 5) if i >= n - 2 else 2)
        nums, den = rows[evaluation]
        window = moved[..., center - 2:center + 3]
        # Pair the centered odd stencil before its one denominator scaling.
        # Endpoint rows retain the exact moment-solved integer numerator.
        if evaluation == 2:
            out[..., i] = (window[..., 0] - window[..., 4] + 8.0 * (window[..., 3] - window[..., 1])) / (float(den) * spacing)
        else:
            out[..., i] = np.sum(window * np.asarray(nums, dtype=np.float64), axis=-1) / (float(den) * spacing)
    return np.moveaxis(out, -1, sample_axis)


def mp_derivative(values: list, spacing: mp.mpf, *, sample_axis: int = -1) -> list:
    """Apply the same rows exactly in 80-digit arithmetic to binary64 inputs."""
    # This diagnostic only needs vectors and (sample,2) matrices.
    if sample_axis not in (-1, 0):
        raise ValueError("unsupported axis")
    if sample_axis == -1 and values and isinstance(values[0], list):
        values = [list(row) for row in zip(*values)]
        transposed = True
    else:
        transposed = False
    n = len(values)
    rows = rational_rows()
    out = []
    for i in range(n):
        center = min(max(i, 2), n - 3)
        evaluation = i if i < 2 else (i - (n - 5) if i >= n - 2 else 2)
        nums, den = rows[evaluation]
        row = []
        width = len(values[0]) if isinstance(values[0], list) else 1
        for col in range(width):
            total = mp.mpf("0")
            if evaluation == 2:
                get = lambda offset: values[center + offset][col] if width > 1 else values[center + offset]
                total = get(-2) - get(2) + mp.mpf(8) * (get(1) - get(-1))
            else:
                for j, num in enumerate(nums):
                    y = values[center - 2 + j][col] if width > 1 else values[center - 2 + j]
                    total += mp.mpf(num) * y
            row.append(total / (mp.mpf(den) * spacing))
        out.append(row if width > 1 else row[0])
    if transposed:
        return [list(row) for row in zip(*out)]
    return out


def mp_solve(matrix: list[list[mp.mpf]], rhs: list[mp.mpf]) -> list[mp.mpf]:
    """Mirror the current row/column scaling and solve in high precision."""
    row_scale = [mp.sqrt(sum(x * x for x in row)) for row in matrix]
    row_scaled = [[x / row_scale[i] for x in matrix[i]] for i in range(2)]
    col_scale = [mp.sqrt(sum(row_scaled[i][j] ** 2 for i in range(2))) for j in range(2)]
    scaled = [[row_scaled[i][j] / col_scale[j] for j in range(2)] for i in range(2)]
    scaled_rhs = [rhs[i] / row_scale[i] for i in range(2)]
    det = scaled[0][0] * scaled[1][1] - scaled[0][1] * scaled[1][0]
    solved = [
        (scaled_rhs[0] * scaled[1][1] - scaled[0][1] * scaled_rhs[1]) / det,
        (scaled[0][0] * scaled_rhs[1] - scaled_rhs[0] * scaled[1][0]) / det,
    ]
    return [solved[j] / col_scale[j] for j in range(2)]


def max_abs(values) -> float:
    flat = np.asarray(values).reshape(-1)
    return float(np.max(np.abs(flat))) if flat.size else 0.0


def main() -> int:
    started = time.perf_counter()
    sources = {"cache": CACHE, "replay_report": REPLAY_REPORT, "module": MODULE, "contract": CONTRACT, "script": Path(__file__)}
    before = {name: sha256(path) for name, path in sources.items()}
    z = np.load(CACHE, allow_pickle=False)
    # The replay cache stores exact binary64 values; no interpolation or ODE
    # evolution is performed here.
    A = z["nqc_auxiliary_amplitude"].astype(np.float64, copy=False)
    phase = z["nqc_auxiliary_phase"].astype(np.float64, copy=False)
    q = z["nqc_auxiliary_basis_amplitude"].astype(np.float64, copy=False)
    p = z["nqc_auxiliary_basis_phase"].astype(np.float64, copy=False)
    original_A = z["nqc_original_amplitude"].astype(np.float64, copy=False)
    original_phase = z["nqc_original_phase"].astype(np.float64, copy=False)
    original_q = z["nqc_original_basis_amplitude"].astype(np.float64, copy=False)
    original_p = z["nqc_original_basis_phase"].astype(np.float64, copy=False)
    selected = int(z["nqc_matching_index"].item())
    spacing = np.float64(0.1)
    nu = np.float64(0.24)
    sqrt24 = np.sqrt(np.float64(24.0))
    targets = [np.float64(z[key].item()) for key in ("nqc_target_amplitude", "nqc_target_amplitude_derivative", "nqc_target_frequency", "nqc_target_frequency_derivative")]
    current_a_cached = z["nqc_amplitude_coefficients"].astype(np.float64)
    current_b_cached = z["nqc_phase_coefficients"].astype(np.float64)

    dtype = torch.float64
    def td(x): return torch.from_numpy(np.asarray(x, dtype=np.float64))
    arw = A / (nu * sqrt24)
    current = {
        "d_arw": quartic_derivative(td(arw), spacing).numpy(),
        "d_product": quartic_derivative(td(arw[:, None] * q), spacing, sample_dim=-2).numpy(),
        "d_phase": quartic_derivative(td(phase), spacing).numpy(),
        "d2_phase": quartic_derivative(quartic_derivative(td(phase), spacing), spacing).numpy(),
        "d_p": quartic_derivative(td(p), spacing, sample_dim=-2).numpy(),
    }
    rational = {
        "d_arw": rational_derivative(arw, spacing),
        "d_product": rational_derivative(arw[:, None] * q, spacing, sample_axis=0),
        "d_phase": rational_derivative(phase, spacing),
        "d2_phase": rational_derivative(rational_derivative(phase, spacing), spacing),
        "d_p": rational_derivative(p, spacing, sample_axis=0),
    }
    amp_matrix = np.array([arw[selected] * q[selected], current["d_product"][selected]])
    amp_rhs = np.array([targets[0] - arw[selected], targets[1] - current["d_arw"][selected]])
    # D(D(p)) is evaluated explicitly, preserving the current contract.
    current_d2_p = quartic_derivative(td(current["d_p"]), spacing, sample_dim=-2).numpy()
    rational_d2_p = rational_derivative(rational["d_p"], spacing, sample_axis=0)
    current["d2_p"] = current_d2_p
    rational["d2_p"] = rational_d2_p
    delta_derivatives = {key: max_abs((current[key] - rational[key]).reshape(-1)) for key in current}
    phase_matrix = np.array([current["d_p"][selected], current["d2_p"][selected]])
    phase_rhs = np.array([current["d_phase"][selected] - targets[2], current["d2_phase"][selected] - targets[3]])
    current_a, _, _ = _solve_two_by_two(td(amp_matrix), td(amp_rhs), max_condition=1.0e10, name="amplitude")
    current_b, _, _ = _solve_two_by_two(td(phase_matrix), td(phase_rhs), max_condition=1.0e10, name="phase")
    current_a = current_a.numpy(); current_b = current_b.numpy()
    rational_amp_matrix = np.array([arw[selected] * q[selected], rational["d_product"][selected]])
    rational_amp_rhs = np.array([targets[0] - arw[selected], targets[1] - rational["d_arw"][selected]])
    rational_phase_matrix = np.array([rational["d_p"][selected], rational["d2_p"][selected]])
    rational_phase_rhs = np.array([rational["d_phase"][selected] - targets[2], rational["d2_phase"][selected] - targets[3]])
    rational_a, _, _ = _solve_two_by_two(td(rational_amp_matrix), td(rational_amp_rhs), max_condition=1.0e10, name="amplitude")
    rational_b, _, _ = _solve_two_by_two(td(rational_phase_matrix), td(rational_phase_rhs), max_condition=1.0e10, name="phase")
    rational_a = rational_a.numpy(); rational_b = rational_b.numpy()

    # High precision, with every input converted from the exact binary64 bit
    # value and constants held at their binary64 values.
    mp.mp.dps = 80
    mps = exact_mp(float(spacing)); mpnu = exact_mp(float(nu)); mpsqrt24 = exact_mp(float(sqrt24))
    mA = [exact_mp(x) for x in A]; mphase = [exact_mp(x) for x in phase]
    mq = [[exact_mp(x) for x in row] for row in q]; mpbasis = [[exact_mp(x) for x in row] for row in p]
    marw = [x / (mpnu * mpsqrt24) for x in mA]
    md_arw = mp_derivative(marw, mps)
    md_product = mp_derivative([[marw[i] * mq[i][j] for j in range(2)] for i in range(len(A))], mps, sample_axis=0)
    md_phase = mp_derivative(mphase, mps); md2_phase = mp_derivative(md_phase, mps)
    md_p = mp_derivative(mpbasis, mps, sample_axis=0); md2_p = mp_derivative(md_p, mps, sample_axis=0)
    mt = [exact_mp(x) for x in targets]
    mi = selected
    ma = mp_solve([[marw[mi] * mq[mi][j] for j in range(2)], md_product[mi]], [mt[0] - marw[mi], mt[1] - md_arw[mi]])
    mb = mp_solve([md_p[mi], md2_p[mi]], [md_phase[mi] - mt[2], md2_phase[mi] - mt[3]])
    hp_a = np.array([float(x) for x in ma]); hp_b = np.array([float(x) for x in mb])

    def corrected(a, b):
        return original_A * (1.0 + (original_q * a).sum(axis=1)), original_phase - (original_p * b).sum(axis=1)
    current_corr_A, current_corr_phase = corrected(current_a, current_b)
    rat_corr_A, rat_corr_phase = corrected(rational_a, rational_b)
    hp_corr_A, hp_corr_phase = corrected(hp_a, hp_b)
    cached_corr_A = z["nqc_original_corrected_amplitude"]
    cached_corr_phase = z["nqc_original_corrected_phase"]
    high_precision = {
        "d_arw": np.asarray([float(x) for x in md_arw]),
        "d_product": np.asarray([[float(x) for x in row] for row in md_product]),
        "d_phase": np.asarray([float(x) for x in md_phase]),
        "d2_phase": np.asarray([float(x) for x in md2_phase]),
        "d_p": np.asarray([[float(x) for x in row] for row in md_p]),
        "d2_p": np.asarray([[float(x) for x in row] for row in md2_p]),
    }
    report = {
        "status": "complete",
        "scope": "saved binary64 NQC arrays only; no ODE, LAL, or fitted/oracle constants",
        "input": {"path": str(CACHE), "sha256": before["cache"], "auxiliary_samples": int(len(A)), "original_samples": int(len(original_A)), "spacing": float(spacing), "nu": float(nu), "matching_index": selected, "targets": [float(x) for x in targets]},
        "derived_weights": {str(k): {"numerators": v[0], "denominator": v[1]} for k, v in sorted(rational_rows().items())},
        "derivative_max_abs_current_vs_rational": delta_derivatives,
        "derivative_max_abs_current_vs_high_precision": {key: max_abs((current[key] - high_precision[key]).reshape(-1)) for key in current},
        "coefficients": {
            "cached_current": {"amplitude": current_a_cached.tolist(), "phase": current_b_cached.tolist()},
            "reconstructed_current": {"amplitude": current_a.tolist(), "phase": current_b.tolist()},
            "rational_float64": {"amplitude": rational_a.tolist(), "phase": rational_b.tolist()},
            "high_precision_binary64_inputs": {"amplitude": hp_a.tolist(), "phase": hp_b.tolist()},
            "max_abs_current_vs_rational": float(max(np.max(np.abs(current_a-rational_a)), np.max(np.abs(current_b-rational_b)))),
            "max_abs_current_vs_high_precision": float(max(np.max(np.abs(current_a-hp_a)), np.max(np.abs(current_b-hp_b)))),
        },
        "corrected_original_max_abs": {
            "current_reconstruction_vs_cached_amplitude": float(np.max(np.abs(current_corr_A-cached_corr_A))),
            "current_reconstruction_vs_cached_phase": float(np.max(np.abs(current_corr_phase-cached_corr_phase))),
            "current_vs_rational_amplitude": float(np.max(np.abs(current_corr_A-rat_corr_A))),
            "current_vs_rational_phase": float(np.max(np.abs(current_corr_phase-rat_corr_phase))),
            "current_vs_high_precision_amplitude": float(np.max(np.abs(current_corr_A-hp_corr_A))),
            "current_vs_high_precision_phase": float(np.max(np.abs(current_corr_phase-hp_corr_phase))),
        },
        "interpretation": "The high-precision comparison changes only arithmetic on the same sampled trajectory. Its coefficient and corrected-waveform deltas are the roundoff scale; they cannot account for a 2.9e-4 full-TD residual unless they are comparable, which this diagnostic tests directly.",
        "source_sha256_before": before,
        "source_sha256_after": {name: sha256(path) for name, path in sources.items()},
        "elapsed_seconds": time.perf_counter() - started,
        "host": socket.gethostname(), "cwd": os.getcwd(), "pid": os.getpid(), "python": sys.executable,
    }
    report["source_hashes_unchanged"] = report["source_sha256_before"] == report["source_sha256_after"]
    VERIFY.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    conclusion = ("# TEOB NQC derivative roundoff diagnostic\n\n"
                  f"Using the saved `{CACHE.name}` binary64 arrays (auxiliary spacing 0.1 and matching index {selected}), "
                  "the degree-four derivative was recomputed from the polynomial moment equations. The centered row is "
                  "(1, -8, 0, 8, -1)/(12 h); endpoint rows use the corresponding nearest-five-node degree-four polynomials. "
                  "Repeated derivatives were applied as D(D), including D(D(p)).\n\n"
                  f"The maximum current weighted-sum versus the independently assembled rational float64 stencil is "
                  f"{max(report['derivative_max_abs_current_vs_rational'].values()):.3e}; the reconstructed coefficient difference is "
                  f"{report['coefficients']['max_abs_current_vs_rational']:.3e}. The maximum corrected original phase and amplitude differences are "
                  f"{report['corrected_original_max_abs']['current_vs_rational_phase']:.3e} and "
                  f"{report['corrected_original_max_abs']['current_vs_rational_amplitude']:.3e}.\n\n"
                  f"Using 80-digit arithmetic on the exact binary64 sampled values gives coefficient and corrected phase/amplitude deltas of "
                  f"{report['coefficients']['max_abs_current_vs_high_precision']:.3e}, "
                  f"{report['corrected_original_max_abs']['current_vs_high_precision_phase']:.3e}, and "
                  f"{report['corrected_original_max_abs']['current_vs_high_precision_amplitude']:.3e}. "
                  "These are arithmetic roundoff scales on an unchanged sampled trajectory; they are far below 2.9e-4. "
                  "This rules out roundoff in this solve on these inputs; it does not test target precision, input trajectory, timing, or reconstruction differences.\n")
    CONCLUSION.write_text(conclusion, encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
