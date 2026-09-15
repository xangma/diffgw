"""Generate compact provenance for the opt-in rounded-knot QNM convention."""
import hashlib, json
from pathlib import Path
import torch
from torchwave.approximants.kerr_qnm_table_convention import KNOTS, rounded_knot_values

ROOT = Path(__file__).resolve().parents[1]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

out = {
    "kind": "diagnostic_qnm_table_convention",
    "grid_provenance": "black-box numerical interpolation metadata; not published or calibration data",
    "value_generation": "fresh signed_qnm_220 recomputed at every knot, each component rounded with Python .6g",
    "spline": "Torch float64 natural cubic, zero endpoint second derivatives",
    "knots": list(KNOTS),
    "rounded_values": rounded_knot_values().tolist(),
    "validation_spins": [-.9231, -.4732, -.0213, .2841, .5763, .8123, .9641, .9971],
    "high_spin_note": "spins >= .9992 can expose numerical source-solver error; no correction is copied here",
    "source_sha256": {
        "table_module": sha(ROOT / "src/torchwave/approximants/kerr_qnm_table_convention.py"),
        "qnm_generator": sha(ROOT / "src/torchwave/approximants/kerr_qnm_generated.py"),
        "generator": sha(Path(__file__)),
    },
}
# Public validation is black-box only and is kept out of the backend.  The
# API returns angular frequency in 1/s for a one-solar-mass final mass.
try:
    import lal
    import lalsimulation as ls
    from torchwave.approximants.kerr_qnm_generated import signed_qnm_220
    from torchwave.approximants.kerr_qnm_table_convention import signed_qnm_220_table_convention
    spins = out["validation_spins"]
    public = []
    for spin in spins:
        vec = lal.CreateCOMPLEX16Vector(1)
        ls.SimIMREOBGenerateQNMFreqV2fromFinal(vec, 1.0, float(spin), 2, 2, 1)
        z = complex(vec.data[0]) * lal.MTSUN_SI
        ref = (z.real, abs(z.imag))
        got = signed_qnm_220(torch.tensor(spin, dtype=torch.float64))
        tab = signed_qnm_220_table_convention(torch.tensor(spin, dtype=torch.float64))
        public.append({"spin": spin, "public_Mfinal_omega": list(ref),
                       "high_precision_error_abs": [abs(float(got[i])-ref[i]) for i in range(2)],
                       "table_error_abs": [abs(float(tab[i])-ref[i]) for i in range(2)]})
    out["public_blackbox_validation"] = public
    out["public_max_abs_error"] = [max(r["table_error_abs"][i] for r in public) for i in range(2)]
    out["public_units"] = "v.data[0] * lal.MTSUN_SI = M_final*omega (angular, dimensionless)"
    out["versions"] = {"lal": lal.__version__, "lalsimulation": ls.__version__, "torch": torch.__version__}
except ImportError as exc:
    out["public_blackbox_validation"] = None
    out["public_validation_error"] = str(exc)
Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
print(json.dumps({"knots": len(KNOTS), "output": str(Path(__file__).with_suffix('.json'))}))
