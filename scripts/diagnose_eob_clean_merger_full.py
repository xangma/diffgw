"""Full-grid clean attachment diagnostic using the published attachment API."""

import hashlib, json, pathlib
import numpy as np, torch
from torchwave.constants import C_SI, MPC_SI
from torchwave.approximants.eob.attachment import attach_seobnrv4_22
import torchwave.approximants.eob.attachment as attachment_module
from torchwave.approximants.eob.nqc import NQCCoefficients
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency
from torchwave.approximants.eob.nqc import seobnrv4_nqc_targets
from torchwave.approximants.eob.remnant import (
    eob_final_mass_fraction,
    hofmann_final_spin,
)
from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm

MTSUN = 4.925490947641267e-6
MSEC = 50 * MTSUN
SCALE = MSEC * C_SI / (100 * 3.0856775814913673e22)
OUT = pathlib.Path(
    "docs/reviews/2026-09-13-parity-review/verification/eob/nqc-merger-full-grid.json"
)


def frequency_rms(x, b, t):
    dt = np.diff(t)
    keep = (
        (np.abs(x[1:]) > np.finfo(float).tiny)
        & (np.abs(x[:-1]) > np.finfo(float).tiny)
        & (np.abs(b[1:]) > np.finfo(float).tiny)
        & (np.abs(b[:-1]) > np.finfo(float).tiny)
    )
    fx = -np.angle(x[1:] * np.conj(x[:-1])) / dt
    fb = -np.angle(b[1:] * np.conj(b[:-1])) / dt
    return {
        "count": int(keep.sum()),
        "rms": float(np.sqrt(np.mean((fx[keep] - fb[keep]) ** 2))),
    }


def metric(x, b):
    phase_mask = np.abs(b) > np.finfo(float).tiny
    ph = np.angle(x[phase_mask] * np.conj(b[phase_mask]))
    return {
        "count": int(len(x)),
        "raw_l2": float(np.linalg.norm(x - b) / np.linalg.norm(b)),
        "phase_rms": float(np.sqrt(np.mean(ph**2))) if ph.size else None,
        "amplitude_ratio_mean": (
            float(np.mean(np.abs(x[phase_mask]) / np.abs(b[phase_mask])))
            if ph.size
            else None
        ),
    }


def controls(
    attach_fn, tin, state, omega, base, freq, nu, chi, sigma, match, targets, dtype
):
    hs = (1e-3, 1e-4, 1e-5)
    out = {}
    for h in hs:
        q = torch.tensor([match - h, match, match + h], dtype=dtype)
        a = attach_fn(
            torch.from_numpy(tin), state, omega, base, freq, nu, chi, sigma, q
        )
        mode = a.mode.detach().numpy()
        target_amp = float(targets[0])
        target_freq = float(targets[3])
        c0 = abs(mode[1] / target_amp - np.exp(-1j * float(a.match_phase)))
        amp_der = (abs(mode[2]) - abs(mode[0])) / (2 * h)
        ph_der = -np.angle(mode[2] * np.conj(mode[0])) / (2 * h)
        out[str(h)] = {
            "c0_relative": float(c0),
            "amplitude_derivative": float(amp_der),
            "target_amplitude_derivative": float(targets[1]),
            "frequency_from_phasor": float(ph_der),
            "target_frequency": target_freq,
            "frequency_error": float(ph_der - target_freq),
        }
    return out


def main():
    d = torch.float64
    nu = torch.tensor(0.24, dtype=d)
    z = torch.tensor(0.0, dtype=d)
    ham = SEOBNRv4Hamiltonian(nu, z, z, delta=torch.tensor(0.2, dtype=d))
    rr = SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8")
    targets = seobnrv4_nqc_targets(nu, z)
    mf = eob_final_mass_fraction(
        torch.tensor(0.6, dtype=d), torch.tensor(0.4, dtype=d), z, z
    )
    af = hofmann_final_spin(
        torch.tensor(0.6, dtype=d), torch.tensor(0.4, dtype=d), z, z, fit_order=(3, 4)
    )
    qw, qd = dimensionless_qnm(2, 2, 0, af)
    sigma = torch.complex(qw / mf, -qd / mf)
    rows = {}
    coeff_path = pathlib.Path(
        "docs/reviews/2026-09-13-parity-review/verification/eob/nqc-analytic-clean.json"
    )
    analytic_report = json.load(open(coeff_path))
    coeff_sha = hashlib.sha256(coeff_path.read_bytes()).hexdigest()
    for label in ("dt_0p5M", "dt_1_4096s"):
        path = pathlib.Path("/tmp/torchwave-public-eob-" + label + ".npz")
        data = np.load(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        post = np.asarray(data["postpeak_state"], dtype=np.float64)
        tin = post[0]
        state = torch.from_numpy(post[1:].T).to(d)
        omega = SEOBNRv4PublishedFlux.from_hamiltonian(state, ham)["omega"]
        base, freq = published_mode22_and_frequency(
            state,
            ham,
            rr,
            spin_v6="none",
            include_pan_v8=True,
            include_positive_chi_540=True,
        )
        out_t = (
            np.arange(len(data["mode22"]), dtype=np.float64)
            * float(data["dt_s"])
            / MSEC
        )
        attached = attach_seobnrv4_22(
            torch.from_numpy(tin),
            state,
            omega,
            base,
            freq,
            nu,
            z,
            sigma,
            torch.from_numpy(out_t),
        )
        match = float(attached.match_time)
        peak_time = float(attached.peak.time)
        baseline_mode = np.asarray(attached.mode.numpy(), dtype=np.complex128) * SCALE
        baseline_controls = controls(
            attach_seobnrv4_22,
            np.asarray(tin),
            state,
            omega,
            base,
            freq,
            nu,
            z,
            sigma,
            match,
            targets,
            d,
        )
        analytic = analytic_report["results"][label]
        coefficient_match = float(analytic["match_time_M"])
        match_delta = match - coefficient_match
        if abs(match_delta) > 2e-6:
            raise RuntimeError(f"{label}: stale coefficient match ({match_delta} M)")
        aa = torch.tensor(analytic["analytic_amplitude"], dtype=d)
        bb = torch.tensor(analytic["analytic_phase"], dtype=d)
        original_calibrate = attachment_module.calibrate_nqc_22
        attachment_module.calibrate_nqc_22 = lambda *a, **k: original_calibrate(
            *a, **k
        )._replace(coefficients=NQCCoefficients(aa, bb))
        try:
            analytic_attached = attach_seobnrv4_22(
                torch.from_numpy(tin),
                state,
                omega,
                base,
                freq,
                nu,
                z,
                sigma,
                torch.from_numpy(out_t),
            )
        finally:
            attachment_module.calibrate_nqc_22 = original_calibrate
        attachment_module.calibrate_nqc_22 = lambda *a, **k: original_calibrate(
            *a, **k
        )._replace(coefficients=NQCCoefficients(aa, bb))
        try:
            analytic_controls = controls(
                attach_seobnrv4_22,
                np.asarray(tin),
                state,
                omega,
                base,
                freq,
                nu,
                z,
                sigma,
                match,
                targets,
                d,
            )
        finally:
            attachment_module.calibrate_nqc_22 = original_calibrate
        mode = np.asarray(analytic_attached.mode.numpy(), dtype=np.complex128) * SCALE
        binary = np.asarray(data["mode22"], dtype=np.complex128)
        m = out_t >= match
        bands = {}
        for lo, hi in (
            (match, float(tin[-1])),
            (match, match + 20),
            (match + 20, match + 50),
            (match + 50, match + 100),
            (match, float(out_t[-1]) + 1),
        ):
            q = m & (out_t >= lo) & (out_t < hi)
            bands[f"{lo:.6f}:{hi:.6f}"] = (
                metric(mode[q], binary[q]) if q.any() else None
            )
        rows[label] = {
            "npz_sha256": sha,
            "input_support_M": [float(tin[0]), float(tin[-1])],
            "output_support_M": [float(out_t[0]), float(out_t[-1])],
            "match_time_M": match,
            "coefficient_match_time_M": coefficient_match,
            "match_consistency_delta_M": match_delta,
            "peak_time_M": peak_time,
            "orbital_peak_frequency": float(attached.peak.frequency),
            "c0_relative_nearest_output_sample": float(
                abs(
                    mode[np.searchsorted(out_t, match)] / (float(targets[0]) * SCALE)
                    - np.exp(-1j * float(attached.match_phase))
                )
            ),
            "frequency_rms_postmatch": frequency_rms(mode[m], binary[m], out_t[m]),
            "baseline_frequency_rms_postmatch": frequency_rms(
                baseline_mode[m], binary[m], out_t[m]
            ),
            "postmatch_count": int(m.sum()),
            "bands": bands,
            "baseline_controls": baseline_controls,
            "analytic_controls": analytic_controls,
            "baseline_bands": {"full_postmatch": metric(baseline_mode[m], binary[m])},
            "baseline_coefficients": {
                "amplitude": attached.calibration.coefficients.amplitude.tolist(),
                "phase": attached.calibration.coefficients.phase.tolist(),
            },
            "analytic_coefficients": {"amplitude": aa.tolist(), "phase": bb.tolist()},
            "remnant_mass_fraction": float(mf),
            "remnant_spin": float(af),
            "sigma_real_imag": [float(sigma.real), float(sigma.imag)],
            "attachment_fields": list(attached._fields),
            "excess_attached_energy_fraction": float(
                np.sum(np.abs(mode[~(np.abs(binary) > np.finfo(float).tiny)]) ** 2)
                / np.sum(np.abs(mode) ** 2)
            ),
        }
    report = {
        "provenance": "Published attach_seobnrv4_22 called directly with paired public postpeak state/time/base mode/base frequency and full binary output grid. No manual phase/amplitude assembly, fitting, or shifts.",
        "script_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes()
        ).hexdigest(),
        "coefficient_report_sha256": coeff_sha,
        "dependency_sha256": {
            p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
            for p in (
                "src/torchwave/approximants/eob/merger_ringdown.py",
                "src/torchwave/approximants/eob/attachment.py",
            )
        },
        "results": rows,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
