"""Read-only LAL binary component diagnostics; no native coefficients are fitted."""
import argparse
import json, math, platform
from pathlib import Path
import numpy as np
import torch
import lal
import lalsimulation as ls
from torchwave.constants import MSUN_SEC
from torchwave.approximants.kerr_ringdown import evaluate_qnm
from torchwave.approximants.kerr_angular import spheroidal_coefficients, spin_weighted_spheroidal_harmonic
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--json-out', type=Path, required=True)
args=parser.parse_args()
args.json_out.parent.mkdir(parents=True,exist_ok=True)
result={'scope':'Independent binary component diagnostics; not full-waveform certification', 'lal':lal.__version__,'lalsimulation':ls.__version__,'host':platform.node(),'frequency_api':'SimIMREOBGenerateQNMFreqV2fromFinal; mass in solar masses, positive damping imaginary part','angular_api':'SimBlackHoleRingdownModeEigenvaluesLeaver and SimBlackHoleRingdownSpheroidalWaveFunctionLeaver', 'frequency_cases':[], 'angular_cases':[], 'unsupported_frequency_cases':[]}
for ell,m in [(2,2),(2,1),(3,3),(3,2),(4,4)]:
 for spin in [0.,.3,.7,.95,.99]:
  reference=lal.CreateCOMPLEX16Vector(3)
  try:
   ls.SimIMREOBGenerateQNMFreqV2fromFinal(reference,60.,spin,ell,m,3)
  except RuntimeError as exc:
   result['unsupported_frequency_cases'].append(dict(ell=ell,m=m,spin=spin,error=str(exc)))
   continue
  for n in range(3):
   w,tau,_=evaluate_qnm(ell,m,n,60.,spin)
   native=complex(w.item(),1/tau.item()); target=reference.data[n]
   result['frequency_cases'].append(dict(ell=ell,m=m,n=n,spin=spin,native=[native.real,native.imag],reference=[target.real,target.imag],relative_complex_error=abs(native-target)/abs(target)))
  if ell!=2 or m!=2 or spin==.99: continue
  A,w2=ls.SimBlackHoleRingdownModeEigenvaluesLeaver(spin/2,ell,m,-2)
  c=spin*w2/2
  native_a,_=spheroidal_coefficients(ell,m,c)
  theta=torch.linspace(.05,math.pi-.05,81,dtype=torch.float64)
  native=spin_weighted_spheroidal_harmonic(ell,m,c,theta).numpy()
  reference=np.array([ls.SimBlackHoleRingdownSpheroidalWaveFunctionLeaver(math.cos(t),spin/2,ell,m,-2,A,w2) for t in theta.numpy()])/math.sqrt(2*math.pi)
  # This overlap only quantifies the arbitrary constant eigenfunction phase;
  # it is not inserted into native code, data, or acceptance gates.
  overlap=np.vdot(native,reference); phase=overlap/abs(overlap)
  result['angular_cases'].append(dict(ell=ell,m=m,spin=spin,A_native=[native_a.real.item(),native_a.imag.item()],A_reference=[A.real,A.imag],A_abs_error=abs(native_a.item()-A),relative_complex_l2=float(np.linalg.norm(native-reference)/np.linalg.norm(reference)),relative_l2_after_constant_phase=float(np.linalg.norm(native*phase-reference)/np.linalg.norm(reference)),reference_constant_phase_rad=float(np.angle(phase)),normalization='LAL integral over cos(theta) normalized to 1; divided by sqrt(2*pi) for native sphere normalization'))
args.json_out.write_text(json.dumps(result,indent=2)+'\n')
print('Frequency max relative complex error',max(x['relative_complex_error'] for x in result['frequency_cases']))
print('Frequency modes at spin .7',[(x['ell'],x['m'],x['n'],x['relative_complex_error']) for x in result['frequency_cases'] if x['spin']==.7])
print('Angular',[(x['spin'],x['A_abs_error'],x['relative_l2_after_constant_phase']) for x in result['angular_cases']])
