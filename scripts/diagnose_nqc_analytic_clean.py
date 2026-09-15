"""Compare clean analytic NQC jets with the published local polynomial solve.

The saved trajectory uses its recorded match_time.  Public post-peak vectors use
an independently refined quartic maximum of clean Hamiltonian omega, then the
published Delta_t.  No waveform oracle or fitted coefficient is used.
"""
import hashlib
import json
import pathlib
import numpy as np
import torch

from torchwave.approximants.eob.nqc import (
    nqc_22_basis, seobnrv4_nqc_targets, seobnrv4_nqc_time_offset,
    calibrate_nqc_22,
)
from torchwave.approximants.eob.nqc_analytic_clean import solve_nqc
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator

SAVED = pathlib.Path("docs/reviews/2026-09-13-parity-review/verification/eob-full20-dt025/full20.pt")
OUT = pathlib.Path("docs/reviews/2026-09-13-parity-review/verification/eob/nqc-analytic-clean.json")


def local_poly(x, y, q, width=5):
    n = width
    degree = width - 1
    i = int(np.searchsorted(x, q))
    center = max(n // 2, min(len(x) - n // 2 - 1, i))
    lo = center - n // 2
    xx = x[lo:lo + n]
    out = []
    for k in range(y.shape[1]):
        origin = xx[n // 2]
        coef = np.polynomial.polynomial.polyfit(xx - origin, y[lo:lo + n, k], degree)
        out.append(np.polynomial.polynomial.polyval(q - origin, coef))
    return np.asarray(out)


def peak_quartic(time, omega):
    i = int(np.argmax(omega))
    if i < 3 or i > len(time) - 4:
        raise ValueError("quartic peak stencil is unsupported")
    lo, hi = i - 3, i + 4
    origin = time[i]
    coeff = np.polynomial.polynomial.polyfit(time[lo:hi] - origin, omega[lo:hi], 4)
    roots = np.roots([4 * coeff[4], 3 * coeff[3], 2 * coeff[2], coeff[1]])
    valid = [float(root.real) for root in roots
             if abs(root.imag) < 1e-8 and time[lo] - origin <= root.real <= time[hi - 1] - origin]
    if not valid:
        return float(time[i])
    return float(origin + min(valid, key=lambda x: abs(x)))


def main():
    dtype = torch.float64
    saved = torch.load(SAVED, weights_only=True)
    saved_time = np.asarray(saved["trajectory_time"].numpy(), dtype=np.float64)
    saved_state = np.asarray(saved["state"].numpy(), dtype=np.float64)
    if "match_time" not in saved:
        raise KeyError("saved trajectory has no match_time metadata")
    saved_match = float(saved["match_time"])

    nu = torch.tensor(0.24, dtype=dtype)
    chi = torch.tensor(0.0, dtype=dtype)
    zero = torch.tensor(0.0, dtype=dtype)
    ham = SEOBNRv4Hamiltonian(nu, zero, zero, delta=torch.tensor(0.2, dtype=dtype))
    radiation = SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8")
    vector_field = EOBTrajectoryIntegrator(ham, radiation).vector_field
    targets = seobnrv4_nqc_targets(nu, chi)
    offset = float(seobnrv4_nqc_time_offset(nu, chi))

    def values(state):
        mode, frequency = published_mode22_and_frequency(
            state.unsqueeze(0), ham, radiation, spin_v6="none",
            include_pan_v8=True, include_positive_chi_540=True,
        )
        omega = SEOBNRv4PublishedFlux.from_hamiltonian(state.unsqueeze(0), ham)["omega"][0]
        # nqc_22_basis returns the three amplitude and two phase bases.
        amp_basis, phase_basis = nqc_22_basis(state[0], state[2], omega)
        amp = mode.abs()[0]
        return torch.stack((amp, amp * amp_basis[0], amp * amp_basis[1],
                            amp * amp_basis[2], phase_basis[0], phase_basis[1], frequency[0]))

    def local_coefficients(state, time, match, width):
        state_t = torch.from_numpy(state).to(dtype)
        mode, frequency = published_mode22_and_frequency(
            state_t, ham, radiation, spin_v6="none",
            include_pan_v8=True, include_positive_chi_540=True,
        )
        omega = SEOBNRv4PublishedFlux.from_hamiltonian(state_t, ham)["omega"].detach().numpy()
        amplitude = mode.abs().detach().numpy()
        gw_frequency = frequency.detach().numpy()
        j = int(np.searchsorted(time, match))
        lo, hi = max(0, j - 3), min(len(time), j + 4)
        if hi - lo != 7:
            raise ValueError("seven-point calibration stencil unsupported")
        result = calibrate_nqc_22(*[torch.from_numpy(np.asarray(a)).to(dtype) for a in
                                  (time[lo:hi], state[lo:hi, 0], state[lo:hi, 2],
                                   omega[lo:hi], amplitude[lo:hi], gw_frequency[lo:hi])],
                                torch.tensor(match, dtype=dtype), targets, degree=6)
        return result.coefficients.amplitude, result.coefficients.phase

    def one(name, state, time, match, width=5):
        if not (time[0] <= match <= time[-1]):
            raise ValueError(f"{name}: match outside support")
        point = local_poly(time, state, match, width).astype(np.float64)
        analytic_amp, analytic_phase, jets = solve_nqc(
            values, torch.from_numpy(point).to(dtype), vector_field, targets)
        sampled_amp, sampled_phase = local_coefficients(state, time, match, width)
        sampled_amp = sampled_amp.detach().numpy()
        sampled_phase = sampled_phase.detach().numpy()
        value, first, second = jets
        return {
            "match_time_M": float(match), "interp_width": int(width),
            "analytic_amplitude": analytic_amp.detach().numpy().tolist(),
            "analytic_phase": analytic_phase.detach().numpy().tolist(),
            "sampled_amplitude": np.asarray(sampled_amp).tolist(),
            "sampled_phase": np.asarray(sampled_phase).tolist(),
            "jet_value": value.numpy().tolist(),
            "jet_first": first.numpy().tolist(),
            "jet_second": second.numpy().tolist(),
            "coefficient_difference": {
                "amplitude": (analytic_amp.detach().numpy() - sampled_amp).tolist(),
                "phase": (analytic_phase.detach().numpy() - sampled_phase).tolist(),
            },
        }

    def rk4_step(state, step):
        k1 = vector_field(state)
        k2 = vector_field(state + 0.5 * step * k1)
        k3 = vector_field(state + 0.5 * step * k2)
        k4 = vector_field(state + step * k3)
        return state + (step / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def rk4_jets(state, step, radius=4):
        center = torch.from_numpy(state).to(dtype)
        points = [center]
        left = center
        right = center
        backward = []
        forward = []
        for _ in range(radius):
            left = rk4_step(left, -step)
            right = rk4_step(right, step)
            backward.append(left)
            forward.append(right)
        points = list(reversed(backward)) + [center] + forward
        sample = torch.stack([values(q) for q in points]).detach().numpy()
        grid = np.arange(-radius, radius + 1, dtype=np.float64) * step
        coeff = np.polynomial.polynomial.polyfit(grid, sample, 4)
        return coeff[0], coeff[1], 2.0 * coeff[2]

    def jet_check(state, step):
        point = torch.from_numpy(state).to(dtype)
        _, _, analytic_jets = solve_nqc(values, point, vector_field, targets)
        value, first, second = (x.detach().numpy() for x in analytic_jets)
        rv, rd, rdd = rk4_jets(state, step)
        return {"step_M": step, "value_max_abs": float(np.max(np.abs(rv-value))),
                "first_max_abs": float(np.max(np.abs(rd-first))),
                "second_max_abs": float(np.max(np.abs(rdd-second)))}

    rows = {"saved": one("saved", saved_state, saved_time, saved_match)}
    rows["saved_rk4_jet_checks"] = [jet_check(local_poly(saved_time, saved_state, saved_match), h)
                                     for h in (0.02, 0.01, 0.005)]
    rows["saved_width_stability"] = {str(width): one("saved", saved_state, saved_time, saved_match, width)
                                     for width in (5, 7, 9)}
    public_meta = {}
    for label in ("dt_0p5M", "dt_1_4096s"):
        data = np.load(f"/tmp/torchwave-public-eob-{label}.npz")
        post = np.asarray(data["postpeak_state"], dtype=np.float64)
        time = post[0]
        state = post[1:].T
        omega = SEOBNRv4PublishedFlux.from_hamiltonian(torch.from_numpy(state).to(dtype), ham)["omega"].numpy()
        peak = peak_quartic(time, omega)
        match = peak + offset
        public_meta[label] = {"peak_time_M": peak, "offset_M": offset}
        rows[label] = one(label, state, time, match)
        rows[label + "_rk4_jet_checks"] = [jet_check(local_poly(time, state, match), h)
                                             for h in (0.02, 0.01, 0.005)]
        rows[label + "_width_stability"] = {str(width): one(label, state, time, match, width)
                                             for width in (5, 7, 9)}

    report = {
        "provenance": "Clean D/D2 includes vector-field derivatives via autograd. Saved match is metadata; public peak is a quartic refinement of clean Hamiltonian omega. Local coefficients are the published seven-point polynomial solve at the identical match state. No oracle fitting.",
        "saved_sha256": hashlib.sha256(SAVED.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "public_peak_metadata": public_meta,
        "results": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
