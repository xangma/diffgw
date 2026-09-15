#!/usr/bin/env python3
"""Black-box diagnostics for the external SEOBNRv4 ROM.

References use only lalsimulation's public binary API. The ringdown input is
the real frequency returned by its public QNM API, expressed in dimensionless
Mf units; it is never fitted to the comparison waveform.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import lal, lalsimulation as ls, numpy as np, torch
from torchwave.approximants.rom.bspline import evaluate_stitched_frequency, load_official_rom

MSUN_SEC = 4.925490947641267e-6

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def reference(case):
    m1, m2, s1, s2 = case
    hp, hc = ls.SimInspiralChooseFDWaveform(
        m1*lal.MSUN_SI, m2*lal.MSUN_SI, 0, 0, s1, 0, 0, s2,
        500*lal.PC_SI*1e6, 0, 0, 0, 0, 0, 1, 20, 512, 0, None,
        ls.GetApproximantFromString("SEOBNRv4_ROM"))
    return np.arange(hp.data.length)*hp.deltaF, hp.data.data.copy(), hc.data.data.copy(), m1+m2

def metrics(mf, out, ref, grids, f_ref_mf):
    valid = (abs(out) > 0) & (abs(ref) > 0)
    valid_order = np.flatnonzero(valid)[np.argsort(mf[valid])]
    phase_all = np.zeros_like(mf, dtype=float)
    phase_sorted = np.unwrap(np.angle(out[valid_order] * np.conj(ref[valid_order])))
    phase_sorted -= phase_sorted[np.argmin(abs(mf[valid_order] - f_ref_mf))]
    phase_all[valid_order] = phase_sorted
    def one(mask):
        mask = mask & valid
        if not mask.any(): return {"count": 0}
        ar = abs(ref[mask])
        order = np.argsort(mf[mask])
        mf_sorted = mf[mask][order]
        phase = phase_all[mask][order]
        # Affine removal is diagnostic only; it is not applied at runtime.
        residual = phase - np.polyval(np.polyfit(mf_sorted, phase, 1), mf_sorted)
        return {"count": int(mask.sum()),
                "amplitude_relative_rms": float(np.linalg.norm(abs(out[mask])-ar)/np.linalg.norm(ar)),
                "phase_raw_rms": float(np.sqrt(np.mean(phase**2))),
                "phase_affine_removed_rms_diagnostic_only": float(np.sqrt(np.mean(residual**2))),
                "complex_relative_rms": float(np.linalg.norm(out[mask][order]-ref[mask][order])/np.linalg.norm(ref[mask]))}
    return {"all": one(valid), "bands": {"low": one(mf <= .009), "transition": one((mf > .009)&(mf < .011)), "high": one(mf >= .011)}, "phase_gauge": {"reference_Mf": float(f_ref_mf), "affine_removal": "diagnostic only"}}

def run_case(case, low, high):
    f, refp, refc, mass = reference(case); eta = case[0]*case[1]/mass**2; mf = f*mass*MSUN_SEC
    mask = (mf >= float(low.mf_grid_amp[0])) & (mf <= float(high.mf_grid_amp[-1]))
    qnm = lal.CreateCOMPLEX16Vector(1)
    ls.SimIMREOBGenerateQNMFreqV2(qnm, case[0], case[1], [0.,0.,case[2]], [0.,0.,case[3]], 2, 2, 1, ls.GetApproximantFromString("SEOBNRv4"))
    # SimIMREOBGenerateQNMFreqV2 returns angular frequency (rad/s).
    rd_raw = float(qnm.data[0].real * mass * MSUN_SEC / (2*np.pi))
    if not (float(high.mf_grid_phase[0]) <= rd_raw <= float(high.mf_grid_phase[-1])):
        raise ValueError(f"public QNM Mf={rd_raw} is outside the high-ROM phase domain")
    rd = rd_raw
    reference_mf = 20 * mass * MSUN_SEC
    amp, phase = evaluate_stitched_frequency(low, high, torch.as_tensor(mf[mask], dtype=torch.float64), eta, case[2], case[3], transition=.01, overlap=(.0074,.014), ringdown_frequency=rd, reference_mf=reference_mf)
    mode = torch.polar(amp, phase).detach().numpy()
    scale = (mass*MSUN_SEC)*(lal.C_SI*mass*MSUN_SEC)/(500*lal.PC_SI*1e6)*.5
    return {"inputs": {"m1_msun":case[0], "m2_msun":case[1], "chi1":case[2], "chi2":case[3], "distance_mpc":500., "ringdown_Mf_qnm_public_api":rd, "ringdown_Mf_qnm_raw":rd_raw, "ringdown_qnm_unit_conversion":"SimIMREOBGenerateQNMFreqV2 real output rad/s divided by 2*pi", "phase_reference_hz":20.}, "metrics":metrics(mf[mask], mode*scale, refp[mask], (low.mf_grid_amp, high.mf_grid_amp), reference_mf)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", type=Path, default=Path("/private/tmp/torchwave-oracle-data/SEOBNRv4ROM_v3.0.hdf5"))
    parser.add_argument("--output", type=Path, default=Path("scripts/diagnose_official_rom.json"))
    args = parser.parse_args(); before = sha256(args.rom); script_path = Path(__file__)
    # Point the public binary backend at the same explicitly supplied asset.
    os.environ["LAL_DATA_PATH"] = str(args.rom.parent)
    low, high = load_official_rom(args.rom, "sub1"), load_official_rom(args.rom, "sub2")
    cases = [(50.,25.,.2,.1), (35.,35.,-.4,.3), (80.,20.,.6,-.2)]
    native = [Path("src/torchwave/approximants/rom/bspline.py"), Path("src/torchwave/approximants/rom/official.py")]
    before_native = {str(p): sha256(p) for p in native}
    result = {"source": {"rom":str(args.rom), "rom_sha256_before":before, "native_hashes_before":before_native, "script_sha256_before":sha256(script_path), "versions":{"python":sys.version, "torch":torch.__version__, "lal":getattr(lal, "__version__", "unknown"), "lalsimulation":getattr(ls, "__version__", "unknown")}}, "method":"Purrer et al. arXiv:1512.02248 IV.1 cubic overlap stitch; public lalsimulation binary reference; QNM from SimIMREOBGenerateQNMFreqV2(SEOBNRv4)", "paper_recipe":{"v2_section":"1512.02248 IV.1: phase cubic fits use Mf=[0.0074,0.014] and 15 sparse points; the high fit is corrected by value and slope at Mf_m=0.01", "v4_section":"1611.03703 §VII: the v4 ROM follows the ROM construction of Refs. 8/117, including patching introduced in Ref. 8, and joins low/high at Mf=0.01; its published general high-spin patch has 149 frequency points while this v3.0 asset has 325. The default all-node v3 fit is provisional metadata handling, not a claimed paper recipe or oracle-selected choice", "sources":["https://arxiv.org/abs/1512.02248","https://arxiv.org/abs/1611.03703"]}, "cases":[run_case(c,low,high) for c in cases]}
    result["source"].update({"rom_sha256_after":sha256(args.rom), "native_hashes_after":{str(p): sha256(p) for p in native}, "script_sha256_after":sha256(script_path)})
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n"); print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__": main()
