"""Compare saved published 22 mode to public LAL EOB mode epochs."""
import hashlib, json, pathlib
import lal, lalsimulation as ls
import numpy as np
import torch
from verify_eob_full20_saved import interp_local

MTSUN = 4.925490947641267e-6
MSEC = 50 * MTSUN
ROOT = pathlib.Path("docs/reviews/2026-09-13-parity-review/verification/eob")
OUT = ROOT / "eob22-public-mode-epoch-fine-quintic.json"
SAVED = pathlib.Path("docs/reviews/2026-09-13-parity-review/verification/eob-full20-dt025/full20.pt")
SAVED_SHA = "56cf48952800928ec3b4037052a9d4010000d841fd90ef497da31e9632d5fd2f"


def run(dt_m):
    dt = dt_m * MSEC
    v = lal.CreateREAL8Vector(0); d = lal.CreateDict()
    out = ls.SimIMRSpinAlignedEOBModes(
        dt, 30*lal.MSUN_SI, 20*lal.MSUN_SI, 20., 100e6*lal.PC_SI,
        0., 0., 4, 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., v, 0, d, d)
    s = out[0]
    while s and (s.l, s.m) != (2, 2):
        s = s.next
    assert s
    return float(s.mode.epoch), dt, np.asarray(s.mode.data.data, dtype=np.complex128)


def main():
    p = torch.load(SAVED, weights_only=True)
    assert hashlib.sha256(SAVED.read_bytes()).hexdigest() == SAVED_SHA
    native_initial = np.asarray(p["time"].numpy()) * MSEC
    native_peak = (np.asarray(p["time"].numpy()) - float(p["peak_time"])) * MSEC
    raw_native_mode = np.asarray(p["mode"].numpy(), dtype=np.complex128)
    mode_scale = MSEC * 299792458 / (100 * 3.085677581491367e22)
    native_mode = raw_native_mode * mode_scale
    # Explicitly test the unprojected-mode SI conversion used below.
    np.testing.assert_allclose(native_mode, raw_native_mode * mode_scale, rtol=0.0, atol=0.0)
    assert mode_scale > 0.0 and np.linalg.norm(native_mode) > 0.0
    assert np.isfinite(native_initial).all() and np.isfinite(native_mode).all()
    assert np.all(np.diff(native_initial) > 0) and np.all(np.diff(native_peak) > 0)
    epoch, dt, binary_mode = run(.5)
    binary_initial = np.arange(len(binary_mode)) * dt
    binary_peak = epoch + binary_initial
    rows = {}
    for name, tn, tb in (("initial", native_initial, binary_initial),
                         ("orbital_peak_centered", native_peak, binary_peak)):
        lo, hi = max(tn[0], tb[0]), min(tn[-1], tb[-1])
        q = tb[(tb >= lo) & (tb <= hi)]
        assert len(q) and q[0] >= lo and q[-1] <= hi
        assert tn[0] <= q[0] <= q[-1] <= tn[-1]
        assert tb[0] <= q[0] <= q[-1] <= tb[-1]
        native_q = interp_local(tn, native_mode, q, 5)
        binary_q = binary_mode[(tb >= lo) & (tb <= hi)]
        assert np.isfinite(native_q).all() and np.isfinite(binary_q).all()
        phase = np.angle(native_q * np.conj(binary_q))
        active = (abs(native_q) > .001*abs(native_q).max()) & (abs(binary_q) > .001*abs(binary_q).max())
        checkpoints = {}
        for point in (0., 100., 1000., 4000., 4900.):
            point_s = point * MSEC
            if name == "initial" and lo <= point_s <= hi:
                j = min(int(np.searchsorted(q, point_s)), len(q)-1)
                checkpoints[str(point)] = {"raw_abs": float(abs(native_q[j]-binary_q[j])), "phase_rad": float(phase[j])}
        rows[name] = {"native_support_M": [float(tn[0]/MSEC), float(tn[-1]/MSEC)],
                      "binary_support_M": [float(tb[0]/MSEC), float(tb[-1]/MSEC)],
                      "common_support_M": [float(lo/MSEC), float(hi/MSEC)], "samples": len(q),
                      "phase_rms_rad": float(np.sqrt(np.mean(phase[active]**2))),
                      "phase_max_abs_rad": float(np.max(abs(phase[active]))), "active_samples": int(active.sum()),
                      "raw_relative_l2": float(np.linalg.norm(native_q-binary_q)/np.linalg.norm(binary_q)),
                      "residual_checkpoints_initial_M": checkpoints}
    report = {"provenance":"Public lalsimulation SimIMRSpinAlignedEOBModes, SEOBNRv4 version 4, nonspinning 30+20 Msun, fMin=20Hz. Fine saved mode is compared at dt=0.5M in initial coordinates and orbital-peak-centered coordinates against the public binary epoch; quintic interpolation is used only within common support, with no optimized shift. The saved unprojected 22 mode is converted to SI strain by mode * (50 M_sun seconds) * c / (100 Mpc), with positive real scale and an exact conversion assertion. The two time-coordinate comparisons are distinct; no causal interpretation of the epoch is assumed.", "saved_sha256":hashlib.sha256(SAVED.read_bytes()).hexdigest(), "saved_sha256_expected":SAVED_SHA, "interpolation":"quintic (reviewed local helper)", "deltaT_M":.5, "deltaT_s":dt, "binary_epoch_s":epoch, "mode_scale":"MSEC*c/(100 Mpc)", "mode_scale_value":mode_scale, "results":rows, "script_sha256":hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    OUT.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__": main()
