"""Independent parent numerical comparison; not a waveform oracle."""
import json, time
from pathlib import Path
import numpy as np
import torch
from torchwave.approximants.polynomial_interval import polynomial_has_nonpositive
rng=np.random.default_rng(20260913)
rows=[]
for degree in range(6):
 for _ in range(100):
  c=rng.normal(size=degree+1)
  critical=np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(c)) if degree else []
  x=[0.,1.]+[float(z.real) for z in critical if abs(z.imag)<1e-10 and 0<z.real<1]
  minimum=min(np.polynomial.polynomial.polyval(x,c))
  if abs(minimum)<1e-9:continue
  padded=np.pad(c,(0,5-degree))
  rows.append((padded,minimum<=0))
c=torch.tensor(np.stack([r[0] for r in rows]),dtype=torch.float64)
start=time.perf_counter()
actual=polynomial_has_nonpositive(c,torch.tensor(0.,dtype=torch.float64),torch.tensor(1.,dtype=torch.float64))
seconds=time.perf_counter()-start
expected=torch.tensor([r[1] for r in rows])
assert torch.equal(actual.cpu(),expected),torch.nonzero(actual.cpu()!=expected)
result=dict(seed=20260913,cases=len(rows),passed=True,batch_seconds=seconds,reference='NumPy derivative roots plus endpoints, excluding minima within 1e-9 of zero; numerical cross-check, not exact certification.')
Path(__file__).with_name('polynomial-parent-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
