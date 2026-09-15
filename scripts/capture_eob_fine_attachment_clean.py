#!/usr/bin/env python3
"""Capture fine attachment intermediates directly from the clean SEOB API."""
import argparse, hashlib, json, math, os, platform, shlex, sys, time
from pathlib import Path
import numpy as np
import torch
from torchwave.approximants.eob.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
from torchwave.constants import MSUN_SEC

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = [ROOT / "src/torchwave/approximants/eob/adaptive_driver_clean.py",
           ROOT / "src/torchwave/approximants/eob/numerical_attachment.py"]
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def a(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output-dir", type=Path, required=True)
    out = ap.parse_args().output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    m1, m2, f, dt = 30., 20., 20., .5
    omega = math.pi*(m1+m2)*float(MSUN_SEC)*f
    startup = {"host": platform.node(), "cwd": str(Path.cwd()), "pid": os.getpid(),
               "command": shlex.join([sys.executable,*sys.argv]),
               "runtime_sha256_before": {str(p):h(p) for p in RUNTIME}}
    (out/"startup.json").write_text(json.dumps(startup, indent=2)+"\n")
    t0=time.monotonic()
    result = reconstruct_seobnrv4_22_adaptive(torch.tensor(omega,dtype=torch.float64),m1,m2,
        0.,0.,dt_geom=dt,num_steps=5000,ringdown_duration=150.,atol=1e-10,rtol=1e-9,
        max_rejected_steps=10000,max_evaluations=100000,max_accepted_steps=5000,max_duration=10000.)
    fine, att = result.trajectory.fine, result.attachment.fine
    nqc = att.nqc
    payload = {
      "fine_output_time":a(fine.output_time), "fine_output_state":a(fine.output_state),
      "fine_output_valid":a(fine.output_valid), "fine_input_time":a(fine.time),
      "fine_input_state":a(fine.state), "fine_input_valid":a(fine.valid),
      # Exact raw fine base reconstructed from the returned full-series
      # amplitude and unwrapped waveform phase (same published mode chart).
      "fine_base_mode":a(nqc.sampled_amplitude)*np.exp(-1j*a(nqc.sampled_waveform_phase)),
      "corrected_inspiral":a(att.corrected_inspiral), "attached_mode":a(att.mode),
      "sampled_amplitude":a(nqc.sampled_amplitude), "sampled_products":a(nqc.sampled_products),
      "sampled_phase_basis":a(nqc.sampled_phase_basis), "sampled_waveform_phase":a(nqc.sampled_waveform_phase),
      "sampled_corrected_amplitude":a(nqc.sampled_corrected_amplitude), "sampled_corrected_phase":a(nqc.sampled_corrected_phase),
      "match_time":a(att.match_time), "boundary_time":a(att.boundary_time),
      "boundary_index":a(att.boundary_index), "boundary_phase":a(att.boundary_phase),
      "boundary_frequency":a(att.boundary_frequency), "boundary_amplitude":a(att.boundary_amplitude),
      "restart_index":np.asarray(result.trajectory.restart_index), "restart_time":a(result.trajectory.restart_time),
      "refinement_factor":np.asarray(result.attachment.refinement_factor), "elapsed_seconds":np.asarray(time.monotonic()-t0),
    }
    np.savez(out/"fine-attachment-intermediates.npz", **payload)
    report={"provenance":"Direct fields from reconstruct_seobnrv4_22_adaptive result.attachment.fine; no cache-writer reattachment.",
      "parameters":{"mass1":m1,"mass2":m2,"f_lower_Hz":f,"dt_geom":dt,"omega_start":omega},
      "fields":sorted(payload),"shapes":{k:list(v.shape) for k,v in payload.items()},
      "sha256":{"script":h(Path(__file__)) ,"npz":h(out/"fine-attachment-intermediates.npz")},"startup":startup}
    (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
if __name__ == "__main__": main()
