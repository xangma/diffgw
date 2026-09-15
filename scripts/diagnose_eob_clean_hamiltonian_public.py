"""Compare clean-Hamiltonian omega on public EOB dynamics at two sample steps.

The LAL call and auxiliary-vector schema are used as black boxes.  Rows of
vector1 are the observed (t,r,phi,prstar,p_phi) schema; this is reported as
an inference, not a public API guarantee.  All comparisons stay in dimensionless
initial-time total-M coordinates and use local quintic interpolation only
inside common support.
"""
import hashlib
import json
import pathlib
import platform

import lal
import lalsimulation as ls
import numpy as np
import torch

from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from verify_eob_full20_saved import interp_local

MTSUN = 4.925490947641267e-6
MSEC = 50.0 * MTSUN
SAVED = pathlib.Path("docs/reviews/2026-09-13-parity-review/verification/eob-full20-dt025/full20.pt")
OUT = pathlib.Path("docs/reviews/2026-09-13-parity-review/verification/eob/eob-clean-hamiltonian-public.json")
VEC_DIR = pathlib.Path("/tmp")


def public_call(dt_s):
    v = lal.CreateREAL8Vector(0)
    d = lal.CreateDict()
    out = ls.SimIMRSpinAlignedEOBModes(
        dt_s, 30 * lal.MSUN_SI, 20 * lal.MSUN_SI, 20.0,
        100e6 * lal.PC_SI, 0.0, 0.0, 4,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, v, 0, d, d)
    series, vector1, vector2 = out
    node = series
    while node and (node.l, node.m) != (2, 2):
        node = node.next
    assert node is not None
    state = np.asarray(vector1.data.data, dtype=np.float64).reshape(5, -1).copy()
    postpeak = np.asarray(vector2.data.data, dtype=np.float64).reshape(5, -1).copy()
    mode = np.asarray(node.mode.data.data, dtype=np.complex128).copy()
    assert state.shape[0] == 5 and np.isfinite(state).all()
    assert np.all(np.diff(state[0]) > 0)
    assert postpeak.shape[0] == 5 and np.isfinite(postpeak).all()
    assert np.all(np.diff(postpeak[0]) > 0)
    assert np.isfinite(postpeak).all() and np.isfinite(mode).all()
    return state, postpeak, mode, float(node.mode.epoch)


def clean_omega(state):
    ham = SEOBNRv4Hamiltonian(
        torch.tensor(0.24, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64), delta=torch.tensor(0.2, dtype=torch.float64))
    x = torch.from_numpy(state[1:].T.copy())
    with torch.inference_mode():
        # from_hamiltonian is a stateless clean adapter for the conservative
        # frequency; no radiation-reaction truncation is needed here.
        omega = SEOBNRv4PublishedFlux.from_hamiltonian(x, ham)["omega"]
    return omega.numpy().astype(np.float64)


def peak_quartic(t, omega):
    i = int(np.argmax(omega))
    if i < 3 or i > len(t) - 4:
        return None
    lo, hi = max(0, i - 3), min(len(t), i + 4)
    coeff = np.polynomial.polynomial.polyfit(t[lo:hi] - t[i], omega[lo:hi], 4)
    roots = np.roots([4 * coeff[4], 3 * coeff[3], 2 * coeff[2], coeff[1]])
    real = [float(x.real) for x in roots if abs(x.imag) < 1e-8 and t[lo] - t[i] <= x.real <= t[hi - 1] - t[i]]
    tp = t[i] if not real else t[i] + real[int(np.argmax([np.polynomial.polynomial.polyval(x, coeff) for x in real]))]
    return float(tp)


def compare(t, state, omega, saved):
    ts = saved["trajectory_time"]
    ss = saved["state"]
    os = saved["omega"]
    lo, hi = max(float(t[0]), float(ts[0])), min(float(t[-1]), float(ts[-1]))
    mask = (t >= lo) & (t <= hi)
    q = t[mask]
    assert len(q) > 100 and hi - lo > 100
    assert ts[0] <= q[0] <= q[-1] <= ts[-1]
    got_state = state[1:, mask].T
    ref_state = np.column_stack([interp_local(ts, ss[:, j], q, 5).real for j in range(4)])
    ref_omega = interp_local(ts, os.astype(complex), q, 5).real
    ds = got_state - ref_state
    do = omega[mask] - ref_omega
    checkpoints = {}
    for point in (0.0, 1000.0, 3000.0, 4500.0, 4900.0):
        if lo <= point <= hi:
            j = min(int(np.searchsorted(q, point)), len(q) - 1)
            checkpoints[str(point)] = {
                "state_abs": np.abs(ds[j]).tolist(), "omega_abs": float(abs(do[j])),
                "public_state": got_state[j].tolist(), "saved_state_interp": ref_state[j].tolist(),
                "public_clean_omega": float(omega[mask][j]), "saved_omega_interp": float(ref_omega[j]),
            }
    return {
        "support_M": [float(lo), float(hi)], "samples": int(len(q)),
        "state_columns": ["r", "phi", "prstar", "p_phi"],
        "state_relative_l2": (np.linalg.norm(ds, axis=0) / np.linalg.norm(ref_state, axis=0)).tolist(),
        "omega_relative_l2": float(np.linalg.norm(do) / np.linalg.norm(ref_omega)),
        "omega_max_abs": float(np.max(abs(do))), "checkpoints_M": checkpoints,
    }


def main():
    expected = "56cf48952800928ec3b4037052a9d4010000d841fd90ef497da31e9632d5fd2f"
    saved_raw = torch.load(SAVED, weights_only=True)
    saved = {"trajectory_time": np.asarray(saved_raw["trajectory_time"].numpy(), dtype=np.float64),
             "state": np.asarray(saved_raw["state"].numpy(), dtype=np.float64),
             "omega": np.asarray(saved_raw["omega"].numpy(), dtype=np.float64)}
    saved_sha = hashlib.sha256(SAVED.read_bytes()).hexdigest()
    assert saved_sha == expected
    results = {}
    for label, dt_s in (("dt_0p5M", 0.5 * MSEC), ("dt_1_4096s", 1.0 / 4096.0)):
        state, postpeak, mode, epoch = public_call(dt_s)
        omega = clean_omega(state)
        postpeak_omega = clean_omega(postpeak)
        assert len(omega) == state.shape[1]
        path = VEC_DIR / ("torchwave-public-eob-" + label + ".npz")
        np.savez(path, state=state, postpeak_state=postpeak, mode22=mode,
                 epoch_s=np.array(epoch), dt_s=np.array(dt_s))
        peak = peak_quartic(state[0], omega)
        postpeak_peak = peak_quartic(postpeak[0], postpeak_omega)
        results[label] = {
            "dt_s": dt_s, "dt_M": dt_s / MSEC, "epoch_s": epoch,
            "length": int(state.shape[1]), "vector_npz": str(path),
            "vector_npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "time_support_M": [float(state[0, 0]), float(state[0, -1])],
            "clean_omega_peak_M": peak,
            "clean_omega_peak_value": None if peak is None else float(np.interp(peak, state[0], omega)),
            "comparison_to_saved": compare(state[0], state, omega, saved),
            "postpeak_time_support_M": [float(postpeak[0, 0]), float(postpeak[0, -1])],
            "postpeak_clean_omega_peak_M": postpeak_peak,
            "postpeak_clean_omega_peak_value": None if postpeak_peak is None else float(np.interp(postpeak_peak, postpeak[0], postpeak_omega)),
            "postpeak_comparison_to_saved": compare(postpeak[0], postpeak, postpeak_omega, saved),
        }
    report = {
        "provenance": "Public lalsimulation SimIMRSpinAlignedEOBModes, SEOBNRv4 v4, 30+20 Msun, nonspinning, fMin=20 Hz. Clean omega is SEOBNRv4PublishedFlux.from_hamiltonian on the observed public vector1 state with float64 nu=.24, chi1=chi2=0, delta=.2. No time/phase optimization.",
        "saved": str(SAVED), "saved_sha256": saved_sha, "saved_sha256_expected": expected,
        "python": "/Users/xangma/miniconda3/envs/pycbcorig313/bin/python",
        "lal_version": getattr(lal, "__version__", "unknown"), "lalsimulation_version": getattr(ls, "__version__", "unknown"),
        "clean_hamiltonian": "SEOBNRv4Hamiltonian(nu=.24, chi1=0, chi2=0, delta=.2); SEOBNRv4PublishedFlux(spin_truncation='published').from_hamiltonian",
        "comparison": "dimensionless initial-time M coordinates; saved trajectory uses reviewed local quintic interpolation only within support; rows=(t,r,phi,prstar,p_phi) are observed schema hypothesis",
        "results": results,
        "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "host": platform.node(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
