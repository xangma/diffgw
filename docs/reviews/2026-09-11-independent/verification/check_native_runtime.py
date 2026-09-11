"""Check frozen catalog cases with LAL imports blocked and no LAL_DATA_PATH."""
import argparse
import importlib.abc
import json
import os
from pathlib import Path
import sys

class BlockLAL(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"lal", "lalsimulation", "lalframe", "lalinspiral", "pycbc"}:
            raise ImportError("LAL/PyCBC disabled for native runtime validation: " + fullname)
        return None

os.environ.pop("LAL_DATA_PATH", None)
sys.meta_path.insert(0, BlockLAL())
import torch
from torchwave import get_fd_waveform

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--json-out", type=Path, required=True)
args = parser.parse_args()
catalog = json.loads(Path(__file__).with_name("strict-catalog-current.json").read_text())
rows = []
for case in catalog["results"]:
    g, p = case["grid"], dict(case["params"])
    p.pop("f_lower", None)
    p.pop("f_final", None)
    f = torch.arange(0.0, g["fhigh"] + g["df"] / 2, g["df"], dtype=torch.float64)
    hp, hc = get_fd_waveform(case["torch_name"], sample_frequencies=f,
        f_lower=g["flow"], f_final=g["fhigh"], dtype=torch.float64, **p)
    assert hp.shape == f.shape and hc.shape == f.shape
    assert torch.isfinite(hp).all() and torch.isfinite(hc).all()
    assert torch.linalg.norm(hp) > 0 and torch.linalg.norm(hc) > 0
    rows.append({"case_id": case["case_id"], "status": "PASSED"})
assert not any(x.split(".")[0] in {"lal", "lalsimulation", "pycbc"} for x in sys.modules)
result = {"note": "Native generation only, not numerical parity; LAL imports blocked, LAL_DATA_PATH unset.", "rows": rows}
args.json_out.write_text(json.dumps(result, indent=2) + "\n")
print(f"{len(rows)} native generation cases passed with LAL imports blocked")
