"""Black-box schema probe for public ``SimIMRSpinAlignedEOBModes``."""
import hashlib, json, pathlib, platform
import lal
import lalsimulation as ls
import numpy as np

PYTHON = "/Users/xangma/miniconda3/envs/pycbcorig313/bin/python"
OUT = pathlib.Path("docs/reviews/2026-09-13-parity-review/verification/eob/lal-modes-api.json")
REFERENCE = pathlib.Path("/tmp/torchwave-lal-eob-modes-reference.npz")


def main():
    dt = 1 / 4096
    nqc = lal.CreateREAL8Vector(0)
    empty = lal.CreateDict()
    invocation = {
        "callable": "lalsimulation.SimIMRSpinAlignedEOBModes",
        "deltaT": dt, "m1_SI": 30 * lal.MSUN_SI, "m2_SI": 20 * lal.MSUN_SI,
        "fMin_Hz": 20., "distance_m": 100e6 * lal.PC_SI,
        "spin1z": 0., "spin2z": 0., "version": 4, "tidals": [0.] * 8,
        "quadrupole_params": [1., 1.], "nqc_length": 0, "nqcFlag": 0,
        "PAPa_dict": "empty", "TGRParams_dict": "empty",
    }
    result = ls.SimIMRSpinAlignedEOBModes(
        dt, invocation["m1_SI"], invocation["m2_SI"], 20., invocation["distance_m"],
        0., 0., 4, 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., nqc, 0, empty, empty)
    series, vector1, vector2 = result
    nodes, modes = [], {}
    node = series
    while node:
        key = f"{node.l}{node.m}"
        values = np.asarray(node.mode.data.data, dtype=np.complex128).copy()
        modes[key] = values
        nodes.append({"l": int(node.l), "m": int(node.m), "name": str(node.mode.name),
                      "length": int(len(values)), "deltaT": float(node.mode.deltaT),
                      "epoch": float(node.mode.epoch), "f0": float(node.mode.f0),
                      "sample_units": str(node.mode.sampleUnits)})
        assert np.isfinite(values).all()
        node = node.next
    v1 = np.asarray(vector1.data.data, dtype=np.float64).copy()
    v2 = np.asarray(vector2.data.data, dtype=np.float64).copy()
    assert np.isfinite(v1).all() and np.isfinite(v2).all()
    assert len(result) == 3 and nodes
    np.savez(REFERENCE, **modes, vector1=v1, vector2=v2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = {"host": platform.node(), "python": PYTHON, "lal_version": getattr(lal, "__version__", "unknown"),
           "lalsimulation_version": getattr(ls, "__version__", "unknown"), "invocation": invocation,
           "returned_type": type(result).__name__, "returned_length": len(result), "mode_nodes": nodes,
           "auxiliary_vectors": [{"length": len(v1), "finite": True}, {"length": len(v2), "finite": True}],
           "reference_npz": str(REFERENCE), "reference_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
           "schema_note": "The public return is [SphHarmTimeSeries, REAL8Vector, REAL8Vector]. The two auxiliary vector semantics are not assigned here because the public callable contract does not name their columns; no trajectory comparison is inferred.",
           "saved_full20_comparison": "Not performed: saved payload columns and auxiliary vector semantics are not identified by this public API contract.",
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    OUT.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
