import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from torchwave.benchmark import benchmark_jaxwave_model, DEFAULT_MODEL_SPECS

for m in ['TaylorF2', 'IMRPhenomD']:
    spec = DEFAULT_MODEL_SPECS.get(m)
    if not spec:
        continue
    for bsz in [1, 10, 100]:
        res = benchmark_jaxwave_model(m, batch_size=bsz, grid=spec['grid'], params=spec['params'], iterations=10, warmup=3)
        if res:
            print(f"[JAX  | CPU  | float64] {m:15s} | B={bsz:<4d} | Med: {res['median_ms']:7.3f} ms | Throughput: {res['wf_per_sec']:8.1f} wf/s | JIT: {res['compile_warmup_ms']:6.1f} ms", flush=True)
