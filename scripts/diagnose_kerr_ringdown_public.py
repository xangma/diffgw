"""Black-box contract diagnostic for LAL's public Kerr ringdown generator.

The callable is intentionally treated as an oracle interface: no LAL source
or inherited TorchWave ringdown body is inspected.  The public output is
compared with a causal damped mode and its FFT using SI inputs.
"""
import hashlib, inspect, json, pathlib
import numpy as np
import torch
import lal, lalsimulation
from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm
from torchwave.approximants.kerr_ringdown_energy import ringdown_amplitude_from_energy_fraction
from torchwave.approximants.kerr_angular import spin_weighted_spheroidal_harmonic

CASES=((65.,.7,100.,0.,0.),(65.,.7,100.,.8,.2),(65.,.7,100.,1.2,.7),(130.,.7,100.,.8,.2),(65.,0.,100.,.8,.2))

def call(mass,spin,distance,inclination,phi,dt=1/8192):
    out=lalsimulation.SimBlackHoleRingdown(lal.LIGOTimeGPS(0),phi,dt,mass*lal.MSUN_SI,spin,.01,distance*1e6*lal.PC_SI,inclination,2,2)
    return [{'length':int(x.data.length),'epoch':float(x.epoch),'deltaT':float(x.deltaT),
             'samples':np.asarray(x.data.data,dtype=float).tolist()} for x in out]

def analytic(mass,spin,distance,inclination,phi,dt,n):
    msec=mass*lal.MTSUN_SI; wr,wi=dimensionless_qnm(2,2,0,spin)
    omega=float(wr)/msec; tau=msec/float(wi)
    amp=float(ringdown_amplitude_from_energy_fraction(mass,distance,.01,omega,tau,convention='single_mode'))
    def pub_s(theta):
        return complex(lalsimulation.SimBlackHoleRingdownSpheroidalWaveFunction(theta,spin,2,2,-2))/np.sqrt(2*np.pi)
    splus,smirror=pub_s(float(inclination)),pub_s(float(np.pi-inclination))
    t=np.arange(n)*dt; z=np.exp(-t/tau-1j*omega*t+1j*2*phi)
    # Published discrete +m/-m pair diagnostic convention.
    hp=-amp/np.sqrt(2)*np.real(z*(splus+smirror))
    hc=amp/np.sqrt(2)*np.imag(z*(splus-smirror))
    return np.stack((hp,hc)), {'amplitude_single_mode':amp,'omega':omega,'tau':tau,
        'splus':[splus.real,splus.imag],'smirror':[smirror.real,smirror.imag]}

def main():
    f=lalsimulation.SimBlackHoleRingdown
    doc=getattr(f,'__doc__',None)
    rows=[]
    for mass,spin,distance,inc,phi in CASES:
        pol=call(mass,spin,distance,inc,phi)
        ana,components=analytic(mass,spin,distance,inc,phi,pol[0]['deltaT'],pol[0]['length'])
        obs=np.array([pol[0]['samples'],pol[1]['samples']])
        time_l2=[float(np.linalg.norm(obs[k]-ana[k])/np.linalg.norm(obs[k])) for k in range(2)]
        fft_l2=[float(np.linalg.norm(np.fft.rfft(obs[k])-np.fft.rfft(ana[k]))/np.linalg.norm(np.fft.rfft(obs[k]))) for k in range(2)]
        rows.append({'mass':mass,'spin':spin,'distance':distance,'inclination':inc,'phi':phi,'polarizations':pol,
                     'array_lengths':[x['length'] for x in pol],
                     'independent_qnm_220': None,'analytic_components':components,'analytic_time_relative_l2':time_l2,'analytic_fft_relative_l2':fft_l2})
    p0=call(65.,.7,100.,.8,0.)
    p90=call(65.,.7,100.,.8,np.pi/2.)
    phase_ratio=[complex(p90[k]['samples'][0],0)/complex(p0[k]['samples'][0],0) for k in range(2)]
    # Keep the independent eigenvalue in ordinary JSON without relying on a
    # private/public LAL QNM implementation.
    zr,zi=dimensionless_qnm(2,2,0,.7)
    for row in rows: row['independent_qnm_220']=[float(zr),float(zi)]
    ang_pub=complex(lalsimulation.SimBlackHoleRingdownSpheroidalWaveFunction(.8,.7,2,2,-2))
    out={'callable':'SimBlackHoleRingdown','docstring':doc,'signature':'LIGOTimeGPS t0, double phi0, double deltaT, double mass, double dimensionless_spin, double fractional_mass_loss, double distance, double inclination, int l, int m','inspect_signature_error':None,
         'cases':rows,'phi_control_0_to_pi_over_2':[[z.real,z.imag] for z in phase_ratio],'fft_comparison':'rfft of complete public and independent analytic time series','source_hash':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'versions':{'lal':lal.__version__,'lalsimulation':lalsimulation.__version__},'clean_room':'public docs/callables plus independent generated Kerr QNM and angular equations; no implementation bodies'}
    out['angular_control']={'public_raw':[ang_pub.real,ang_pub.imag], 'public_div_sqrt_2pi':[ang_pub.real/np.sqrt(2*np.pi),ang_pub.imag/np.sqrt(2*np.pi)]}
    pathlib.Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
if __name__=='__main__': main()
