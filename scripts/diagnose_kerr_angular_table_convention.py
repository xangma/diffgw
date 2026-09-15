"""Black-box public versus clean inferred angular compatibility diagnostic."""
import hashlib,json,sys
from pathlib import Path
import numpy as np, torch
import lalsimulation as ls
from torchwave.approximants.kerr_angular_table_convention import spin_weighted_spheroidal_harmonic_table_compat

spins=[.1,.3,.7,.9,.95]; mu=np.array([1,.8,.4,0,-.4,-.8],float)
rows=[]
for a in spins:
    th=torch.acos(torch.tensor(mu,dtype=torch.float64)); own_full=spin_weighted_spheroidal_harmonic_table_compat(a,th).detach().numpy(); own=own_full/own_full[0]
    A,w=ls.SimBlackHoleRingdownModeEigenvaluesLeaver(a/2,2,2,-2)
    pub_full=np.array([ls.SimBlackHoleRingdownSpheroidalWaveFunctionLeaver(float(x),a/2,2,2,-2,A,w) for x in mu])/np.sqrt(2*np.pi); pub=pub_full/pub_full[0]
    rows.append({'spin':a,'shape_max_abs':float(np.max(abs(own-pub))),
                 'full_complex_max_abs':float(np.max(abs(own_full-pub_full))),
                 'own_abs_at_mu0':float(abs(own_full[3])),
                 'public_A':[A.real,A.imag],'public_omega2':[w.real,w.imag]})
out={'scope':'diagnostic only; no production routing','threshold':0.01,'threshold_rule':'absolute abs(a_n*z^n), stop at first n>=1 below threshold','normalization':'piecewise Gauss-Legendre integral split at inferred threshold breakpoints; no fitting','provenance':'termination and metadata inferred from public black-box outputs, not published calibration','spins':rows,'source_sha256':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [Path(__file__),Path('src/torchwave/approximants/kerr_angular_table_convention.py')]},'versions':{'lalsimulation':ls.__version__,'torch':torch.__version__}}
Path('scripts/diagnose_kerr_angular_table_convention.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({'max_shape_error':max(r['shape_max_abs'] for r in rows),'rows':len(rows)}))
