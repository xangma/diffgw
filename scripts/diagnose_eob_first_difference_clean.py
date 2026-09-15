#!/usr/bin/env python3
"""Black-box public SEOBNRv4 NQC coefficient probe (validation only)."""
import hashlib, json, platform, sys
from pathlib import Path
import lal
import lalsimulation as ls
import numpy as np
import torch
from torchwave.approximants.eob.nqc import seobnrv4_nqc_targets, seobnrv4_nqc_time_offset, seobnrv4_nqc_time_offset_numerical_compatibility, nqc_22_factor
from torchwave.approximants.eob.numerical_attachment import _full_series_nqc
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/reviews/2026-09-13-parity-review/verification/eob-first-difference-clean"
NAMES = ("a1", "a2", "a3", "a3S", "a4", "a5", "b1", "b2", "b3", "b4")

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    dt = .5 * 50 * lal.MTSUN_SI
    coeff = lal.CreateREAL8Vector(10)
    coeff.data[:] = np.nan
    d = lal.CreateDict()
    result = ls.SimIMRSpinAlignedEOBModes(
        dt, 30*lal.MSUN_SI, 20*lal.MSUN_SI, 20., 100e6*lal.PC_SI,
        0., 0., 4, 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., coeff, 1, d, d)
    vals = np.asarray(coeff.data, dtype=np.float64).copy()
    public_peak_hz = float(ls.SimIMRSpinAlignedEOBPeakFrequency(
        30*lal.MSUN_SI, 20*lal.MSUN_SI, 0., 0., 4))
    total_mass_seconds = 50 * lal.MTSUN_SI
    clean_target_frequency = float(seobnrv4_nqc_targets(
        torch.tensor(.24, dtype=torch.float64), torch.tensor(0., dtype=torch.float64))[3])
    public_peak_dimensionless = float(2*np.pi*total_mass_seconds*public_peak_hz)
    cap = ROOT / "docs/reviews/2026-09-13-parity-review/verification/eob-fine-stage-replay-clean/fine-attachment-intermediates.npz"
    with np.load(cap) as c:
        tt = torch.from_numpy(c["fine_output_time"]).double()
        ss = torch.from_numpy(c["fine_output_state"]).double()
        aa = torch.from_numpy(c["sampled_amplitude"]).double()
        pp = torch.from_numpy(c["sampled_waveform_phase"]).double()
        nu = torch.tensor(.24, dtype=torch.float64); zero = torch.tensor(0., dtype=torch.float64)
        ham = SEOBNRv4Hamiltonian(nu, zero, zero, delta=torch.tensor(.2, dtype=torch.float64))
        om = SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8").from_hamiltonian(ss, ham)["omega"]
        clean = _full_series_nqc(tt, ss[:, 0], ss[:, 2], om, aa, om, pp,
                                 torch.as_tensor(c["match_time"]).double(),
                                 seobnrv4_nqc_targets(nu, zero), max_condition=1e10).coefficients
        clean_vals = np.asarray(clean.amplitude.tolist() + [0., 0., 0.] + clean.phase.tolist() + [0., 0.], dtype=float)
        offset = float(seobnrv4_nqc_time_offset(nu, zero))
        # Contract diagnostic: round each printed fit coefficient independently,
        # then evaluate the polynomial (never round the final evaluated sum).
        coeffs = (716.044155, -13.087878, -45.883834, -2.504992)
        rounded_coeffs = tuple(float(format(c, ".6g")) for c in coeffs)
        rounded_offset = rounded_coeffs[0]*.24**3 + rounded_coeffs[1]*.24**2 + rounded_coeffs[2]*.24 + rounded_coeffs[3]
        compatibility_offset = float(seobnrv4_nqc_time_offset_numerical_compatibility(nu, zero))
        assert compatibility_offset == rounded_offset
        rounded_match = torch.as_tensor(c["match_time"]).double() + (rounded_offset - offset)
        rounded = _full_series_nqc(tt, ss[:, 0], ss[:, 2], om, aa, om, pp, rounded_match,
                                   seobnrv4_nqc_targets(nu, zero), max_condition=1e10).coefficients
        rounded_vals = np.asarray(rounded.amplitude.tolist() + [0., 0., 0.] + rounded.phase.tolist() + [0., 0.], dtype=float)
        rounded_mode = bm = torch.from_numpy(c["fine_base_mode"]).to(torch.complex128) * nqc_22_factor(
            ss[:, 0], ss[:, 2], om, (rounded.amplitude, rounded.phase), sample_dims=1)
        paper_mode = torch.from_numpy(c["fine_base_mode"]).to(torch.complex128) * nqc_22_factor(
            ss[:, 0], ss[:, 2], om, (clean.amplitude, clean.phase), sample_dims=1)
        saved_corrected = torch.from_numpy(c["corrected_inspiral"]).to(torch.complex128)
        mode_scale = 50 * lal.MTSUN_SI * lal.C_SI / (100e6 * lal.PC_SI)
        rel = lambda x, y: float(torch.linalg.vector_norm(x-y) / torch.linalg.vector_norm(y))
        public = np.load("/tmp/torchwave-public-eob-dt_0p5M.npz")
        ref = int(c["refinement_factor"]); restart = int(c["restart_index"])
        ai = np.arange(0, len(c["fine_base_mode"]), ref); ni = restart + ai // ref
        boundary_index = int(np.asarray(c["boundary_index"]))
        mask = (ai < boundary_index) & (ni < len(public["mode22"])); ai = ai[mask]; ni = ni[mask]
        public_mode = torch.from_numpy(public["mode22"][ni]).to(torch.complex128)
        public_compare = {"samples": int(mask.sum()), "boundary_index_fine": boundary_index, "fine_index_range": [int(ai[0]), int(ai[-1])], "native_indices": [int(ni[0]), int(ni[-1])], "paper_corrected_vs_public_relative_l2": rel(paper_mode[ai]*mode_scale, public_mode), "rounded_timing_vs_public_relative_l2": rel(rounded_mode[ai]*mode_scale, public_mode), "mask": "fine indices strictly before saved boundary_index, plus public support; exact native mapping restart + fine_index//refinement_factor", "public_fixture": "/tmp/torchwave-public-eob-dt_0p5M.npz", "public_fixture_sha256": sha(Path('/tmp/torchwave-public-eob-dt_0p5M.npz'))}
    series, v1, v2 = result
    nodes = []
    while series:
        nodes.append({"l": int(series.l), "m": int(series.m),
                      "length": len(series.mode.data.data), "epoch_s": float(series.mode.epoch)})
        series = series.next
    def vec_info(v):
        if v is None: return {"returned": None}
        a = np.asarray(v.data, dtype=float)
        return {"returned": True, "length": len(a), "head": a[:8].tolist()}
    payload = {
        "provenance": "Public lalsimulation SimIMRSpinAlignedEOBModes black-box call; coefficients returned through caller-owned REAL8Vector with nqcFlag=1. Values are validation observables only and are not installed or fitted.",
        "invocation": {"m1_msun": 30., "m2_msun": 20., "f_min_hz": 20., "dt_M": .5, "spin1z": 0., "spin2z": 0., "version": 4, "nqcFlag": 1, "nqc_vector_length": 10},
        "coefficient_names": list(NAMES),
        "coefficients": {k: float(v) for k, v in zip(NAMES, vals)},
        "clean_fine_full_series_coefficients": {k: float(v) for k, v in zip(NAMES, clean_vals)},
        "coefficient_delta_public_minus_clean": {k: float(a-b) for k, a, b in zip(NAMES, vals, clean_vals)},
        "peak_frequency_comparison": {"public_peak_frequency_hz": public_peak_hz, "public_dimensionless_2piMf": public_peak_dimensionless, "clean_target_dimensionless_2piMf": clean_target_frequency, "public_minus_clean": public_peak_dimensionless-clean_target_frequency},
        "time_offset_precision_diagnostic": {"original_paper_precision_offset_M": offset, "per_coefficient_six_significant_digit_offset_M": rounded_offset, "nonspinning_offset_coefficients_original": coeffs, "nonspinning_offset_coefficients_rounded_each": rounded_coeffs, "match_time_shift_M": rounded_offset-offset, "rounded_match_coefficients": {k: float(v) for k, v in zip(NAMES, rounded_vals)}, "rounded_minus_original_coefficients": {k: float(a-b) for k, a, b in zip(NAMES, rounded_vals, clean_vals)}},
        "raw_waveform_precision_effect": {"paper_recomputed_vs_saved_corrected_inspiral_relative_l2": rel(paper_mode, saved_corrected), "rounded_vs_paper_corrected_inspiral_relative_l2": rel(rounded_mode, paper_mode), "scale_to_public_100mpc": mode_scale, "scope": "same captured fine state/time, same clean targets, fixed peak, no ODE or alignment"},
        "public_inspiral_comparison": public_compare,
        "interpretation": "First observed mismatch: the public NQC coefficients differ slightly from the clean captured full-series solve, while the public peak-frequency target is compared separately. These measurements do not by themselves distinguish source trajectory, sampled products, target, match time, or derivative-stencil semantics; no correction is justified yet.",
        "auxiliary_vectors": {"vector1": vec_info(v1), "vector2": vec_info(v2)},
        "mode_nodes": nodes,
        "runtime": {"python": sys.executable, "host": platform.node(), "lal_version": getattr(lal, "__version__", "unknown"), "lalsimulation_version": getattr(ls, "__version__", "unknown")},
        "input_hashes": {"fine_capture_npz": sha(cap)},
        "module_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT/"src/torchwave/approximants/eob/nqc.py",
            ROOT/"src/torchwave/approximants/eob/numerical_attachment.py",
            ROOT/"src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
            ROOT/"src/torchwave/approximants/eob/seobnrv4_flux.py")},
        "script_sha256": sha(Path(__file__)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

if __name__ == "__main__": main()
