#!/usr/bin/env python3
"""Replay cached clean attachment data with old/new public NQC offsets."""
import hashlib, json
from pathlib import Path
import numpy as np
import torch

from torchwave.approximants.eob.nqc import (
    seobnrv4_nqc_time_offset, seobnrv4_nqc_time_offset_numerical_compatibility)
from torchwave.approximants.eob.numerical_attachment import attach_seobnrv4_22_numerical_compatibility
from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency
from torchwave.approximants.eob.remnant import eob_final_mass_fraction, hofmann_final_spin
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.kerr_qnm_table_convention import signed_qnm_220_table_convention
from torchwave.constants import C_SI, MPC_SI, MSUN_SEC

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "docs/reviews/2026-09-13-parity-review/verification/eob-fine-stage-replay-clean/fine-attachment-intermediates.npz"
PUBLIC = Path("/tmp/torchwave-public-eob-dt_0p5M.npz")
OUT = CAP.parent / "nqc-offset-precision-replay.json"
CAPTURE = ROOT / "scripts/capture_eob_fine_attachment_clean.py"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def rel(a, b): return float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(b))
def err_ratio(a, b): return float(torch.linalg.vector_norm(a) / torch.linalg.vector_norm(b))

def main():
    c, p = np.load(CAP), np.load(PUBLIC)
    dtype = torch.float64
    time = torch.tensor(c["fine_output_time"], dtype=dtype)
    state = torch.tensor(c["fine_output_state"], dtype=dtype)
    base = torch.tensor(c["fine_base_mode"], dtype=torch.complex128)
    ham = SEOBNRv4Hamiltonian(torch.tensor(.24, dtype=dtype), torch.tensor(0., dtype=dtype), torch.tensor(0., dtype=dtype))
    rr = SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8", retain_pan_cubic_spin=True)
    mode, frequency = published_mode22_and_frequency(
        state, ham, rr, spin_v6="bohe_total_energy", include_pan_v8=False,
        include_positive_chi_540=False, retain_pan_cubic_spin=True)
    quantities = rr.from_hamiltonian(state, ham)
    mf = eob_final_mass_fraction(torch.tensor(.6, dtype=dtype), torch.tensor(.4, dtype=dtype), torch.tensor(0., dtype=dtype), torch.tensor(0., dtype=dtype))
    af = hofmann_final_spin(torch.tensor(.6, dtype=dtype), torch.tensor(.4, dtype=dtype), torch.tensor(0., dtype=dtype), torch.tensor(0., dtype=dtype), fit_order=(3, 4))
    qw, qd = signed_qnm_220_table_convention(af)
    sigma = torch.complex(qw / mf, -qd / mf)
    nu, chi = torch.tensor(.24, dtype=dtype), torch.tensor(0., dtype=dtype)
    new_offset = seobnrv4_nqc_time_offset_numerical_compatibility(nu, chi)
    old_offset = seobnrv4_nqc_time_offset(nu, chi)
    new_match = torch.tensor(float(c["match_time"]), dtype=dtype)
    old_match = new_match
    new_match = old_match + (new_offset - old_offset)
    output_time = torch.arange(c["attached_mode"].size, dtype=dtype) * (time[1] - time[0])
    common = (time, state, quantities["omega"], base, frequency, nu, chi, sigma, output_time)
    old = attach_seobnrv4_22_numerical_compatibility(*common, valid=torch.tensor(c["fine_output_valid"]), match_time=old_match)
    new = attach_seobnrv4_22_numerical_compatibility(*common, valid=torch.tensor(c["fine_output_valid"]), match_time=new_match)
    cached_full = torch.tensor(c["attached_mode"], dtype=torch.complex128)
    old_full_error = old.mode - cached_full
    # Compare only exact fine samples represented on the public grid.
    native = torch.tensor(p["mode22"], dtype=torch.complex128)
    fine_idx = np.arange(0, old.mode.numel(), int(c["refinement_factor"]))
    native_idx = int(c["restart_index"]) + fine_idx // int(c["refinement_factor"])
    keep_np = native_idx < native.numel()
    fine_idx_t = torch.tensor(fine_idx[keep_np], dtype=torch.long)
    indices = torch.tensor(native_idx[keep_np], dtype=torch.long)
    keep = torch.ones_like(indices, dtype=torch.bool)
    # The public waveform is strain at 100 Mpc; attachment.mode is geometric.
    scale = 50 * float(MSUN_SEC) * float(C_SI) / (100 * float(MPC_SI))
    old_s = old.mode[fine_idx_t] * scale
    new_s = new.mode[fine_idx_t] * scale
    ref = native[indices]
    cached = torch.tensor(c["attached_mode"], dtype=torch.complex128)[fine_idx_t] * scale
    report = {
        "provenance": "Cached clean fine trajectory replay; same state, base mode, frequency, remnant/QNM sigma, and exact mapped dt=0.5M public samples; no ODE rerun, alignment, fitting, or LAL source.",
        "inputs_sha256": {str(CAP): sha(CAP), str(PUBLIC): sha(PUBLIC)},
        "module_sha256": {str(ROOT / "src/torchwave/approximants/eob/nqc.py"): sha(ROOT / "src/torchwave/approximants/eob/nqc.py"), str(ROOT / "src/torchwave/approximants/eob/numerical_attachment.py"): sha(ROOT / "src/torchwave/approximants/eob/numerical_attachment.py"), str(ROOT / "src/torchwave/approximants/eob/remnant.py"): sha(ROOT / "src/torchwave/approximants/eob/remnant.py"), str(ROOT / "src/torchwave/approximants/kerr_qnm_table_convention.py"): sha(ROOT / "src/torchwave/approximants/kerr_qnm_table_convention.py")},
        "script_sha256": {str(Path(__file__)): sha(Path(__file__)), str(CAPTURE): sha(CAPTURE)},
        "parameters": {"nu": .24, "chi": 0., "cached_match_time": float(old_match), "old_match_time": float(old_match), "new_match_time": float(new_match), "old_offset": float(old_offset), "new_offset": float(new_offset), "samples_compared": int(keep.sum()), "geometric_to_100mpc_scale": scale, "sigma_real": float(sigma.real), "sigma_imag": float(sigma.imag)},
        "metrics": {"old_vs_cached_attached_relative_l2": err_ratio(old_full_error, cached_full), "old_vs_cached_attached_max_abs": float(torch.max(torch.abs(old_full_error))), "old_vs_public_relative_l2": rel(old_s, ref), "new_vs_public_relative_l2": rel(new_s, ref), "new_minus_old_relative_l2": rel(new_s, old_s), "old_boundary_time": float(old.boundary_time), "new_boundary_time": float(new.boundary_time)},
        "scope": "The old reconstruction is first checked against the saved attached output over its complete array; public comparison then uses exact refinement-grid samples only.",
        "reconstruction": {"old_mode_length": int(old.mode.numel()), "cached_attached_length": int(cached_full.numel())}
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
if __name__ == "__main__": main()
