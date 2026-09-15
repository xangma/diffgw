"""Black-box PhenomD/QNM compatibility diagnostic.

Only public callable outputs are used here.  This is evidence gathering, not
a replacement for the PhenomD fit or a phase correction.
"""
import hashlib
import json
import inspect
import importlib.metadata
from pathlib import Path

import numpy as np
import torch

from torchwave.approximants.imrphenomd import IMRPhenomD, evaluate_kerr_qnm_220


PYTHON = "/Users/xangma/miniconda3/envs/pycbcorig313/bin/python"
_GATE_DIR = Path("docs/reviews/2026-09-13-parity-review/verification/nsbh-amplitude-complete-gates")


def _load_gate_cases():
    """Load the exact public gate inventories; select only BBH PhenomD rows."""
    cases = {}
    for filename in ("catalog.json", "holdout.json"):
        data = json.loads((_GATE_DIR / filename).read_text())
        for row in data["metadata"]["required_inventory"]:
            if row.get("torch_name") == "IMRPhenomD":
                cases[row["id"]] = {"id": row["id"], "grid": row["grid"], "params": row["params"]}
    return cases


def _lal_waveform(f, p):
    import lal
    import lalsimulation as ls
    v = lal.CreateREAL8Vector(len(f)); v.data[:] = f
    m1, m2 = p["mass1"], p["mass2"]
    s1, s2, fref = p["chi1"], p["chi2"], p["f_ref"]
    out = ls.SimIMRPhenomDFrequencySequence(
        v, 0.0, fref, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
        s1, s2, p["distance"] * lal.PC_SI * 1.0e6, None, ls.NoNRT_V)
    return np.asarray(out.data.data, dtype=np.complex128)


def _metrics(native, public):
    finite = np.isfinite(native.real) & np.isfinite(native.imag) & np.isfinite(public.real) & np.isfinite(public.imag)
    if not np.all(finite):
        raise ValueError(f"nonfinite waveform sample(s): {int((~finite).sum())}")
    n, p = native[finite], public[finite]
    raw = np.linalg.norm(n - p) / max(np.linalg.norm(p), 1e-300)
    mag = np.linalg.norm(np.abs(n) - np.abs(p)) / max(np.linalg.norm(np.abs(p)), 1e-300)
    phase_ok = (np.abs(n) > 1e-300) & (np.abs(p) > 1e-300)
    z = np.unwrap(np.angle(n[phase_ok] / p[phase_ok]))
    x = np.arange(z.size, dtype=float)
    if z.size < 2:
        slope = intercept = nonaff = None
    else:
        coeff = np.polyfit(x, z, 1)
        slope, intercept = float(coeff[0]), float(coeff[1])
        nonaff = float(np.linalg.norm(z - np.polyval(coeff, x)) / max(np.sqrt(z.size), 1.0))
    return {"finite_support": int(finite.sum()), "raw_relative_l2": float(raw),
            "phase_support": int(phase_ok.sum()), "magnitude_relative_l2": float(mag),
            "phase_slope_per_bin": slope, "phase_intercept": intercept,
            "phase_nonaffine_rms": nonaff}


def _public_qnm_provider(original, lal, ls):
    """Return original remnant properties with public EOB QNM frequencies.

    The inspected EOB API returns angular frequencies per solar mass.
    Multiplication by ``M_f * lal.MTSUN_SI`` produces the dimensionless
    M_f omega pair expected by evaluate_kerr_qnm_220.
    """
    def provider(mass1, mass2, chi1, chi2):
        base = original(mass1, mass2, chi1, chi2)
        mf, af = base[:2]
        mf_solar = float(mf.detach())
        v = lal.CreateCOMPLEX16Vector(1)
        ls.SimIMREOBGenerateQNMFreqV2fromFinal(v, mf_solar, float(af.detach()), 2, 2, 1)
        # remnant_properties documents M*f (cycles), while evaluate_kerr
        # reports M*omega; convert the public angular frequency by 2*pi.
        # Convert public final-mass M*omega to the remnant_properties
        # total-mass M*f convention.
        mtot = float((mass1 + mass2).detach()) * lal.MTSUN_SI
        q = v.data[0] * mtot / (2.0 * np.pi)
        return (*base[:2], torch.as_tensor(q.real, dtype=mf.dtype, device=mf.device),
                torch.as_tensor(q.imag, dtype=mf.dtype, device=mf.device))
    return provider


def main():
    import lalsimulation as ls
    cases = _load_gate_cases()
    model = IMRPhenomD()
    records = []
    for label, case in cases.items():
        p = case["params"]; grid = case["grid"]
        m1, m2 = p["mass1"], p["mass2"]
        s1, s2, fref = p["chi1"], p["chi2"], p["f_ref"]
        f = np.arange(grid["flow"], grid["fhigh"] + grid["df"] / 2, grid["df"])
        ft = torch.tensor(f, dtype=torch.float64)
        args = (ft, torch.tensor(m1, dtype=torch.float64), torch.tensor(m2, dtype=torch.float64),
                torch.tensor(s1, dtype=torch.float64), torch.tensor(s2, dtype=torch.float64))
        # SimIMRPhenomDFrequencySequence is the intrinsic complex carrier
        # interface (no inclination argument).  Compare the matching native
        # face-on, zero-phase carrier while retaining the gate geometry in
        # the recorded exact inputs.
        hp, hc = model(*args, distance=p["distance"], inclination=0.0,
                       phic=0.0, tc=p["tc"], f_ref=fref, window_type=None)
        native = hp.detach().cpu().numpy()
        provider = _public_qnm_provider(model.remnant_properties, __import__("lal"), ls)
        def identity(a, b, c, d):
            base = model.remnant_properties(a, b, c, d)
            mf0, af0 = base[:2]
            qr, qi = evaluate_kerr_qnm_220(af0)
            total = a + b
            return mf0, af0, qr * total / mf0 / (2.0 * np.pi), qi * total / mf0 / (2.0 * np.pi)
        hp_q, _ = model(*args, distance=p["distance"], inclination=0.0,
                        phic=0.0, tc=p["tc"], f_ref=fref, window_type=None,
                        remnant_provider=provider)
        substituted = hp_q.detach().cpu().numpy()
        hp_i, _ = model(*args, distance=p["distance"], inclination=0.0,
                        phic=0.0, tc=p["tc"], f_ref=fref, window_type=None,
                        remnant_provider=identity)
        public = _lal_waveform(f, p)
        mm = model.remnant_properties(*args[1:])
        mf, af = [float(x.detach()) for x in mm[:2]]
        qre, qim = [float(x.detach()) for x in evaluate_kerr_qnm_220(torch.tensor(af, dtype=torch.float64))]
        vpub = __import__("lal").CreateCOMPLEX16Vector(1)
        ls.SimIMREOBGenerateQNMFreqV2fromFinal(vpub, mf, af, 2, 2, 1)
        qpub = vpub.data[0] * (mf * __import__("lal").MTSUN_SI)
        # TorchWave's public D remnant_properties returns M_f in solar-mass
        # units, whereas the LAL utility returns M_f/(m1+m2).  Normalize only
        # for this comparison; retain the raw value for provenance.
        records.append({"label": label, "inputs": p, "grid": grid, "metrics": _metrics(native, public),
                        "identity_provider_metrics": _metrics(hp_i.detach().cpu().numpy(), native),
                        "public_qnm_substitution_metrics": _metrics(substituted, public),
                        "native_remnant_mass_raw_solar_mass": mf,
                        "native_remnant_mass_fraction": mf / (m1 + m2),
                        "native_remnant_spin": af,
                        "native_qnm_Momega": {"real": qre, "minus_imag": qim},
                        "public_eob_qnm_Momega": {"real": float(qpub.real), "minus_imag": float(qpub.imag)},
                        "public_final_spin": float(ls.SimIMRPhenomDFinalSpin(m1, m2, s1, s2)),
                        "public_final_mass_fraction": float(ls.SimPhenomUtilsIMRPhenomDFinalMass(m1, m2, s1, s2)),
                        "native_phase_derivative": float(model.compute_slope_peak(*args[1:]).detach()),
                        "public_peak_frequency_hz": float(ls.IMRPhenomDGetPeakFreq(m1, m2, s1, s2))})
    src = ["scripts/diagnose_d_qnm_contract.py", "src/torchwave/approximants/imrphenomd.py"]
    hashes = {x: hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in src}
    # The canonical harness supplies the full hp/hc and detector projection
    # metrics.  Preserve its exact gate inputs and status as a paired
    # external-reference record; the provider substitution remains intrinsic
    # above because the harness model dispatch is not monkeypatched globally.
    from parity_harness import run_oracle_comparison
    harness = []
    for case in cases.values():
        hp = dict(case["params"])
        harness_case = {"id": case.get("id", "D_qnm_diagnostic"), "torch_name": "IMRPhenomD",
                        "lal_name": "IMRPhenomD", "params": hp, "grid": case["grid"],
                        "expected_gate": {"complex_relative_l2": 1e-6, "mismatch": 1e-8}}
        harness.append({"mode": "baseline", "result": run_oracle_comparison(harness_case, strict=False)})
    # The harness dispatches through the native class.  Replace only its
    # remnant callable for this process, restoring it unconditionally.
    original = IMRPhenomD.remnant_properties
    try:
        IMRPhenomD.remnant_properties = staticmethod(_public_qnm_provider(original, __import__("lal"), ls))
        for case in cases.values():
            hp = dict(case["params"])
            harness_case = {"id": case.get("id", "D_qnm_diagnostic"), "torch_name": "IMRPhenomD",
                            "lal_name": "IMRPhenomD", "params": hp, "grid": case["grid"],
                            "expected_gate": {"complex_relative_l2": 1e-6, "mismatch": 1e-8}}
            harness.append({"mode": "public_qnm_provider", "result": run_oracle_comparison(harness_case, strict=False)})
    finally:
        IMRPhenomD.remnant_properties = original
    def callable_signature(name):
        try:
            return str(inspect.signature(getattr(ls, name)))
        except (TypeError, ValueError):
            return "builtin signature unavailable"
    public_names = sorted(
        n for n in dir(ls)
        if any(token in n.lower() for token in ("qnm", "ringdown", "phenomd", "alpha"))
    )
    versions = {}
    for package in ("torch", "numpy", "lal", "lalsimulation"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            try:
                module = __import__(package)
                versions[package] = str(getattr(module, "__version__", "module-version-unreported"))
            except Exception:
                versions[package] = "unavailable"
    out = {"metadata": {"python": PYTHON, "versions": versions,
                         "frequency_grid": "gate grid 20..512 Hz, df=.25",
                         "lal_sequence_mode": {"name": "NoNRT_V", "value": int(ls.NoNRT_V),
                                               "meaning": "BBH PhenomD without NRTidal correction"},
                         "source_sha256": hashes,
                         "public_callables": {n: callable_signature(n) if n in dir(ls) else None
                                              for n in ("SimIMRPhenomDFinalSpin", "SimPhenomUtilsIMRPhenomDFinalMass", "IMRPhenomDGetPeakFreq")},
                         "public_component_inventory": {
                             n: callable_signature(n) for n in public_names
                             if callable(getattr(ls, n, None))
                         },
                         "qnm_contract_note": (
                             "No public PhenomD QNM-frequency callable was exposed by the "
                             "inspected lalsimulation API; IMRPhenomDGetPeakFreq is the "
                             "available peak-frequency callable. Native evaluate_kerr_qnm_220 "
                             "is recorded as a TorchWave black-box component."
                         )},
           "records": records, "canonical_harness_full_projection": harness}
    Path("docs/diagnose_d_qnm_contract.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
