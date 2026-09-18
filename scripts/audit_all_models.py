import sys
import warnings
import traceback

import os
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
import torch
import torchwave
from torchwave.waveforms import _APPROXIMANTS

models_to_test = [
    ("taylorf2", dict(mass1=30.0, mass2=20.0, chi1=0.2, chi2=-0.1)),
    ("taylorf2hm", dict(mass1=30.0, mass2=20.0, chi1=0.2, chi2=-0.1)),
    ("taylorf2eccentric", dict(mass1=30.0, mass2=20.0, chi1=0.2, chi2=-0.1, eccentricity=0.05)),
    ("imrphenomd", dict(mass1=30.0, mass2=20.0, chi1=0.2, chi2=-0.1)),
    ("imrphenomp", dict(mass1=30.0, mass2=20.0, spin1z=0.2, spin2z=-0.1)),
    ("imrphenompv2", dict(mass1=30.0, mass2=20.0, spin1z=0.2, spin2z=-0.1)),
    ("imrphenomxp", dict(mass1=30.0, mass2=20.0, chi1=0.2, chi2=-0.1)),
    ("imrphenomdnrtidal", dict(mass1=1.4, mass2=1.3, chi1=0.0, chi2=0.0, lambda1=100.0, lambda2=100.0)),
    ("imrphenomdnrtidalv2", dict(mass1=1.4, mass2=1.3, chi1=0.0, chi2=0.0, lambda1=100.0, lambda2=100.0)),
    ("imrphenomhm", dict(mass1=30.0, mass2=20.0, chi1=0.2, chi2=-0.1)),
    ("imrphenomxhm", dict(mass1=30.0, mass2=20.0, chi1=0.2, chi2=-0.1)),
    ("imrphenomxas", dict(mass1=30.0, mass2=20.0, chi1=0.2, chi2=-0.1)),
    ("imrphenomxphm", dict(mass1=30.0, mass2=20.0, chi1=0.2, chi2=-0.1)),
    ("imrphenomxphmspintaylor", dict(mass1=30.0, mass2=20.0, chi1=0.2, chi2=-0.1)),
    ("imrphenomxpnr", dict(mass1=30.0, mass2=20.0, chi1=0.2, chi2=-0.1)),
    ("imrphenomnsbh", dict(mass1=5.0, mass2=1.4, chi1=0.1, chi2=0.0, lambda2=400.0)),
    ("kerrringdown", dict(mass1=50.0, mass2=10.0, chi1=0.7, chi2=0.0)),
    ("seobnrrom", dict(mass1=30.0, mass2=20.0, chi1=0.2, chi2=-0.1)),
    ("seobnrv5rom", dict(mass1=30.0, mass2=20.0, chi1=0.2, chi2=-0.1)),
    ("teobresums", dict(mass1=30.0, mass2=20.0, chi1=0.2, chi2=-0.1)),
    ("seobnr", dict(mass1=30.0, mass2=20.0, chi1=0.2, chi2=-0.1)),
    ("seobnrv4published", dict(mass1=30.0, mass2=20.0, chi1=0.2, chi2=-0.1)),
    ("seobnrv4publishednonspinning", dict(mass1=30.0, mass2=20.0)),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
freqs = torch.linspace(40.0, 512.0, 32, dtype=torch.float64, device=device)

print(f"{'Approximant':<25} | {'Class':<22} | {'Breaks':<6} | {'Graphs':<6} | {'C-Warns':<7} | Status", flush=True)
print("-" * 90, flush=True)

for key, kw in models_to_test:
    cls_name = _APPROXIMANTS[key].__name__
    common_kw = dict(kw)
    common_kw["distance"] = 100.0
    common_kw["inclination"] = 0.7
    common_kw["sample_frequencies"] = freqs
    
    # 1. Test Eager & catch complex warnings
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            hp, hc = torchwave.get_fd_waveform(approximant=key, **common_kw)
            c_warns = [str(m.message) for m in recorded if "complex" in str(m.message).lower()]
    except Exception as e:
        print(f"{key:<25} | {cls_name:<22} | {'-':<6} | {'-':<6} | {'-':<7} | Eager Failed: {type(e).__name__}: {str(e)[:40]}", flush=True)
        continue

    # 2. Test Compilation & Graph Breaks
    if cls_name in ("SEOBNR_ODE", "SEOBNRv4Published", "SEOBNRv4PublishedNonspinning"):
        print(f"{key:<25} | {cls_name:<22} | {'ODE':<6} | {'ODE':<6} | {len(c_warns):<7} | Dynamic ODE integrator (non-static loop)", flush=True)
        continue

    def fn():
        return torchwave.get_fd_waveform(approximant=key, **common_kw)
    
    try:
        explanation = torch._dynamo.explain(fn)()
        gb = explanation.graph_break_count
        g = explanation.graph_count
        status = "CLEAN (0 breaks)" if gb == 0 else f"{gb} breaks"
        print(f"{key:<25} | {cls_name:<22} | {gb:<6} | {g:<6} | {len(c_warns):<7} | {status}", flush=True)
        if gb > 0:
            seen = set()
            for r in explanation.break_reasons:
                first_line = str(r.reason).split("\n")[0][:75]
                if first_line not in seen:
                    print(f"    * break: {first_line}", flush=True)
                    seen.add(first_line)
    except Exception as e:
        print(f"{key:<25} | {cls_name:<22} | {'-':<6} | {'-':<6} | {len(c_warns):<7} | Explain Failed: {type(e).__name__}: {str(e)[:40]}", flush=True)

print("=" * 90, flush=True)
print("AUDIT COMPLETE", flush=True)
