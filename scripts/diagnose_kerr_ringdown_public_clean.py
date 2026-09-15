"""Clean TD Kerr220 against the public callable (diagnostic only)."""
import hashlib,json
from pathlib import Path
import numpy as np, lal, lalsimulation as ls, torch
from torchwave.approximants.kerr_ringdown_public_clean import kerr220_td
spins=[0.,.1,.3,.7,.9,.95]; rows=[]
for spin in spins:
    dt=1/8192; pub=ls.SimBlackHoleRingdown(lal.LIGOTimeGPS(0),.2,dt,65*lal.MSUN_SI,spin,.01,100e6*lal.PC_SI,.8,2,2)
    n=int(pub[0].data.length); t=torch.arange(n,dtype=torch.float64)*dt
    hp,hc=kerr220_td(t,65.,spin,100.,.01,.8,.2,0.)
    obs=[np.asarray(x.data.data) for x in pub]
    rows.append({'spin':spin,'length':n,'clean_td_relative_l2':[float(np.linalg.norm(obs[k]-v.detach().numpy())/np.linalg.norm(obs[k])) for k,v in enumerate((hp,hc))]})
out={'scope':'diagnostic only; no production routing','cases':rows,'energy_fraction':.01,'angular_convention':'own inferred public-compatible shape; norm is independent piecewise quadrature','source_sha256':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [Path(__file__),Path('src/torchwave/approximants/kerr_ringdown_public_clean.py')]},'versions':{'lal':lal.__version__,'lalsimulation':ls.__version__,'torch':torch.__version__}}
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n'); print(json.dumps(out))
