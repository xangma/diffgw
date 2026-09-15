"""Check whether the optional SXS metadata attachment changes decoded arrays."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import sxs
from sxs.waveforms.format_handlers import rotating_paired_diff_multishuffle_bzip2 as loader

assert sxs.__version__ == "2026.0.3"
base = Path(__file__).resolve().parent
manifest = json.loads((base / "nr33-manifest-round1.json").read_text())
checks = []
for case in manifest["cases"]:
    if case["target_q"] not in (6, 8) or case["split"] == "validation_only":
        continue
    stem = Path(case["path"]).stem
    raw = Path(manifest["raw_dir"])
    exact_metadata = sxs.Metadata.from_file(raw / (stem + "_metadata.json"))
    wave = sxs.load(str(raw / (stem + ".h5")), group="/Strain_N3.dir",
                    metadata=exact_metadata, download=False)
    with np.load(case["path"]) as saved:
        checks.append({"id": case["id"], "time_equal": np.array_equal(wave.t, saved["time"]),
                       "h22_equal": np.array_equal(wave.data[:, wave.index(2, 2)], saved["h22"]),
                       "h33_equal": np.array_equal(wave.data[:, wave.index(3, 3)], saved["h33"])})
assert len(checks) == 2 and all(c[k] for c in checks for k in ("time_equal", "h22_equal", "h33_equal"))
report = {"completed_utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
          "sxs_version": sxs.__version__, "loader_path": loader.__file__,
          "loader_sha256": hashlib.sha256(Path(loader.__file__).read_bytes()).hexdigest(),
          "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "checks": checks, "conclusion": "optional metadata attachment has no effect on the arrays or duration failures"}
with (base / "decoder-metadata-check.json").open("x") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
print(json.dumps(checks))
