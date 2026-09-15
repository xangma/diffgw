#!/usr/bin/env python3
"""Stagewise SEOB comparison on the native cached output grid.

The cache and public binary are both sampled at 0.5M.  This intentionally
compares values at identical timestamps; no public waveform interpolation is
performed.  It is a diagnostic only and never feeds coefficients back into
the runtime.
"""
from pathlib import Path
import json
import argparse
import hashlib
import numpy as np
import torch

from torchwave.approximants.eob.numerical_attachment import attach_seobnrv4_22_numerical_compatibility
from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.eob.remnant import eob_final_mass_fraction, hofmann_final_spin
from torchwave.approximants.kerr_qnm_table_convention import signed_qnm_220_table_convention
from torchwave.constants import C_SI

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "docs/reviews/2026-09-13-parity-review/verification/eob-adaptive-full20-clean/native-full20-controller-clean.npz"
ADAPTIVE_REPORT = ROOT / "docs/reviews/2026-09-13-parity-review/verification/eob-adaptive-full20-clean/diagnostic.json"
PUBLIC = Path("/tmp/torchwave-public-eob-dt_0p5M.npz")
DTYPE = torch.float64
MTSUN_S = 4.925490947641267e-6
MASS_S = 50 * MTSUN_S
SCALE = MASS_S * C_SI / (100 * 3.0856775814913673e22)

def metrics(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.abs(b) > 0
    a, b = a[m], b[m]
    return {"n": int(m.sum()), "complex_rel_l2": float(np.linalg.norm(a-b)/np.linalg.norm(b)),
            "amp_rel_l2": float(np.linalg.norm(np.abs(a)-np.abs(b))/np.linalg.norm(np.abs(b))),
            "phase_rms": float(np.sqrt(np.mean(np.angle(a*np.conj(b))**2)))}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with np.load(CACHE, allow_pickle=False) as d:
        t = d["time"].astype(np.float64); state = d["trajectory_state"].astype(np.float64)
        valid = d["trajectory_valid"].astype(bool); native = d["mode_100mpc"].astype(np.complex128)
    with np.load(PUBLIC, allow_pickle=False) as d: public = d["mode22"].astype(np.complex128)
    n_valid = int(valid.sum()); n = min(n_valid, public.size, t.size); t=t[:n]; state=state[:n]; native_valid=native[:n]; public_valid=public[:n]
    nu=torch.tensor(.24,dtype=DTYPE); zero=torch.tensor(0.,dtype=DTYPE)
    ham=SEOBNRv4Hamiltonian(nu,zero,zero,delta=torch.tensor(.2,dtype=DTYPE)); flux=SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8")
    st=torch.from_numpy(state); tt=torch.from_numpy(t); omega=flux.from_hamiltonian(st,ham)["omega"]
    base, freq=published_mode22_and_frequency(st,ham,flux,spin_v6="bohe_total_energy",include_pan_v8=False,include_positive_chi_540=False)
    mf=eob_final_mass_fraction(torch.tensor(.6,dtype=DTYPE),torch.tensor(.4,dtype=DTYPE),zero,zero); fs=hofmann_final_spin(torch.tensor(.6,dtype=DTYPE),torch.tensor(.4,dtype=DTYPE),zero,zero,fit_order=(3,4)); q,damp=signed_qnm_220_table_convention(fs); sigma=torch.complex(q,-damp)/mf
    out=attach_seobnrv4_22_numerical_compatibility(tt,st,omega,base,freq,nu,zero,sigma,tt)
    base_np=base.detach().numpy()*SCALE; corr_np=out.corrected_inspiral.detach().numpy()*SCALE; att_np=out.mode.detach().numpy()*SCALE
    actual_boundary=float(json.loads(ADAPTIVE_REPORT.read_text())["adaptive"]["boundary_time_M"])
    pre=t < actual_boundary; post=(t >= actual_boundary) & (np.abs(public_valid)>0)
    coarse_pre=t < float(out.boundary_time); coarse_post=(t >= float(out.boundary_time)) & (np.abs(public_valid)>0)
    native_full=np.zeros(public.size,dtype=np.complex128); native_full[:min(native.size,public.size)]=native[:min(native.size,public.size)]
    result={"timestamp_grid": {"n":int(n),"dt_M":float(t[1]-t[0]),"comparison":"cached assembled mode and public mode share the 0.5M grid"},"stages": {"cached_native_assembled_full_padded":metrics(native_full,public),"cached_native_truncated_valid_window":metrics(native_valid,public_valid),"cached_native_pre_boundary_valid_window":metrics(native_valid[pre],public_valid[pre]),"cached_native_post_boundary_valid_window":metrics(native_valid[post],public_valid[post]),"coarse_grid_reattachment_control_full_overlap":metrics(att_np,public_valid),"coarse_grid_reattachment_pre_boundary":metrics(corr_np[coarse_pre],public_valid[coarse_pre]),"coarse_grid_reattachment_post_boundary":metrics(att_np[coarse_post],public_valid[coarse_post])},"events":{"recorded_actual_fine_boundary_time_M":actual_boundary,"coarse_control_boundary_time_M":float(out.boundary_time),"coarse_control_boundary_index":int(out.boundary_index)},"inputs_sha256":{"cache":hashlib.sha256(CACHE.read_bytes()).hexdigest(),"public_fixture":hashlib.sha256(PUBLIC.read_bytes()).hexdigest(),"adaptive_diagnostic":hashlib.sha256(ADAPTIVE_REPORT.read_bytes()).hexdigest(),"script":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},"limitations":["The cached native mode is the actual fine-grid pipeline output; full padded comparison includes its ringdown and zero tail.","The reattachment control intentionally attaches on the downsampled 0.5M trajectory and is not a production-stage estimate.","The cache does not contain fine-grid base and NQC-corrected complex mode arrays, so source versus boundary attribution cannot be isolated without a new fine-grid replay."]}
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
if __name__ == "__main__": main()
