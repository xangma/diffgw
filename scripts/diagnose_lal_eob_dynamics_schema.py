"""Schema and identity diagnostics for public LAL EOB auxiliary vectors."""
import hashlib, json, pathlib, platform
import numpy as np
import torch

REF = pathlib.Path("/tmp/torchwave-lal-eob-modes-reference.npz")
SAVED = pathlib.Path("docs/reviews/2026-09-13-parity-review/verification/eob-full20-dt025/full20.pt")
OUT = pathlib.Path("docs/reviews/2026-09-13-parity-review/verification/eob/lal-eob-dynamics-schema.json")
MTSUN = 4.925490947641267e-6


def stats(a):
    rows = []
    for shape in ((5, -1), (-1, 5)):
        x = a.reshape(shape)
        cols = []
        for j in range(5):
            c = x[j] if shape[0] == 5 else x[:, j]
            cols.append({"first": float(c[0]), "last": float(c[-1]), "min": float(c.min()),
                         "max": float(c.max()), "strict_increasing": bool(np.all(np.diff(c) > 0)),
                         "nondecreasing": bool(np.all(np.diff(c) >= 0))})
        rows.append({"reshape": list(x.shape), "columns": cols})
    return rows


def interp_local(x, y, q, degree=5):
    n = degree + 1
    out = np.empty(q.shape, dtype=y.dtype)
    for k, value in enumerate(q):
        j = int(np.searchsorted(x, value))
        left = max(0, min(len(x) - n, j - n // 2))
        xx = x[left:left+n]; center = xx[n // 2]
        scale = max(abs(xx[-1] - xx[0]), np.finfo(float).eps)
        u = (xx - center) / scale
        out[k] = np.polynomial.polynomial.polyval(
            (value-center)/scale,
            np.polynomial.polynomial.polyfit(u, y[left:left+n], degree))
    return out


def compare(candidate, saved):
    # The observed physically consistent interpretation is rows=(t,r,phi,pr*,p_phi).
    t, state = candidate[0], candidate[1:]
    ts = saved["trajectory_time"]
    lo, hi = max(t[0], ts[0]), min(t[-1], ts[-1])
    q = t[(t >= lo) & (t <= hi)]
    assert len(q) and q[0] >= lo and q[-1] <= hi and np.all(np.diff(q) > 0)
    interp = np.column_stack([interp_local(ts, saved["state"][:, j], q) for j in range(4)])
    got = np.column_stack([state[j][(t >= lo) & (t <= hi)] for j in range(4)])
    diff = got - interp
    checkpoints = {}
    for point in (0., 1000., 3000., 4500., 4900.):
        if lo <= point <= hi:
            ix = int(np.searchsorted(q, point))
            ix = min(ix, len(q)-1)
            checkpoints[str(point)] = np.abs(diff[ix]).tolist()
    return {"common_support_M": [float(lo), float(hi)], "common_support_s": [float(lo*50*MTSUN), float(hi*50*MTSUN)], "samples": int(len(q)),
            "columns": ["r", "phi", "prstar", "pphi"],
            "relative_l2": (np.linalg.norm(diff, axis=0) / np.linalg.norm(interp, axis=0)).tolist(),
            "max_abs": np.max(abs(diff), axis=0).tolist(), "absolute_residual_at_M": checkpoints}


def main():
    z = np.load(REF)
    saved_payloads = {}
    for label, path in (("coarse", pathlib.Path("docs/reviews/2026-09-13-parity-review/verification/eob-full20/full20.pt")),
                        ("fine", SAVED)):
        saved_t = torch.load(path, weights_only=True)
        saved_payloads[label] = {"trajectory_time": np.asarray(saved_t["trajectory_time"].numpy()),
                                 "state": np.asarray(saved_t["state"].numpy()),
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    vectors = {k: np.asarray(z[k], dtype=np.float64) for k in ("vector1", "vector2")}
    inferred = {}
    for name, a in vectors.items():
        x = a.reshape(5, -1)
        if name == "vector1":
            assert len(x[0]) > 4000 and x[0][-1] - x[0][0] > 4900
        inferred[name] = {"length": len(a), "reshape_candidates": stats(a),
                          "observed_physical_schema_rows": ["time", "r", "phi", "prstar", "p_phi"],
                          "physical_identity_checks": {
                              "time_increasing": bool(np.all(np.diff(x[0]) > 0)),
                              "radius_positive": bool(np.all(x[1] > 0)),
                              "phi_increasing": bool(np.all(np.diff(x[2]) > 0)),
                              "prstar_nonpositive": bool(np.all(x[3] <= 0)),
                              "p_phi_positive": bool(np.all(x[4] > 0)),
                          },
                          "saved_trajectory_comparison_initial": {
                              label: compare(x, payload) for label, payload in saved_payloads.items()}}
    out = {"host": platform.node(), "reference_npz": str(REF),
           "reference_sha256": hashlib.sha256(REF.read_bytes()).hexdigest(),
           "saved_input_sha256": {k: v["sha256"] for k, v in saved_payloads.items()},
           "vectors": inferred,
           "interpretation_note": "Rows=(time,r,phi,prstar,p_phi) are an observed physically consistent hypothesis, not a documented LAL guarantee. Auxiliary vector semantics remain absent from the callable docstring.",
           "comparison_note": "Times remain in dimensionless total-M coordinates with initial origin; no event shift or optimization. Saved trajectories are interpolated locally with quintic polynomials only within overlap. The public mode grid has dt=1/4096 s (physical), corresponding to approximately 0.99M here.",
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__": main()
