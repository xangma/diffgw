"""Clean carrier/NQC attribution on saved and public EOB trajectories."""
import hashlib,json,pathlib
import numpy as np,torch
from torchwave.constants import C_SI,MPC_SI
from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency
from torchwave.approximants.eob.attachment import attach_seobnrv4_22
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.nqc import nqc_22_factor
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.eob.remnant import eob_final_mass_fraction,hofmann_final_spin
from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm
from verify_eob_full20_saved import interp_local
MTSUN=4.925490947641267e-6;MSEC=50*MTSUN
SAVED=pathlib.Path('docs/reviews/2026-09-13-parity-review/verification/eob-full20-dt025/full20.pt')
OUT=pathlib.Path('docs/reviews/2026-09-13-parity-review/verification/eob/eob-clean-waveform-attribution.json')

def setup():
 d=torch.float64;nu=torch.tensor(.24,dtype=d);z=torch.tensor(0.,dtype=d)
 ham=SEOBNRv4Hamiltonian(nu,z,z,delta=torch.tensor(.2,dtype=d));rr=SEOBNRv4PublishedFlux(spin_truncation='generic-leading-ell8')
 mf=eob_final_mass_fraction(torch.tensor(.6,dtype=d),torch.tensor(.4,dtype=d),z,z);af=hofmann_final_spin(torch.tensor(.6,dtype=d),torch.tensor(.4,dtype=d),z,z,fit_order=(3,4));qw,qd=dimensionless_qnm(2,2,0,af);sigma=torch.complex(qw/mf,-qd/mf)
 return ham,rr,nu,torch.tensor(0.,dtype=d),sigma

def evaluate(state,time,omega=None,attach=True):
 ham,rr,nu,chi,sigma=setup(); st=torch.from_numpy(np.asarray(state,dtype=np.float64)); tt=torch.from_numpy(np.asarray(time,dtype=np.float64))
 if omega is None: omega=SEOBNRv4PublishedFlux.from_hamiltonian(st,ham)['omega'].numpy()
 with torch.inference_mode(): base,freq=published_mode22_and_frequency(st,ham,rr,spin_v6='none',include_pan_v8=True,include_positive_chi_540=True)
 if not attach: return base.numpy(), None
 with torch.inference_mode(): att=attach_seobnrv4_22(tt,st,torch.from_numpy(omega),base,freq,nu,chi,sigma,tt)
 return base.numpy(),att

def metrics(x,b,tm,points=(0,100,1000,3000,4500,4900)):
 ph=np.angle(x*np.conj(b));out={'raw_relative_l2':float(np.linalg.norm(x-b)/np.linalg.norm(b)),'phase_rms_rad':float(np.sqrt(np.mean(ph**2))),'checkpoints':{}}
 for p in points:
  if tm[0]<=p<=tm[-1]:
   j=min(int(np.searchsorted(tm,p)),len(tm)-1);out['checkpoints'][str(p)]={'sample_M':float(tm[j]),'raw_abs':float(abs(x[j]-b[j])),'phase_rad':float(ph[j])}
 return out

def main():
 p=torch.load(SAVED,weights_only=True);sh=hashlib.sha256(SAVED.read_bytes()).hexdigest();assert sh=='56cf48952800928ec3b4037052a9d4010000d841fd90ef497da31e9632d5fd2f'
 ts=np.asarray(p['trajectory_time'].numpy());ss=np.asarray(p['state'].numpy());so=np.asarray(p['omega'].numpy());st=np.asarray(p['time'].numpy());native=np.asarray(p['mode'].numpy(),dtype=np.complex128)
 scale=MSEC*C_SI/(100*MPC_SI);base_native,att_native=evaluate(ss,ts,so);native_att=att_native.mode.numpy();
 results={}
 for label in ('dt_0p5M','dt_1_4096s'):
  z=np.load('/tmp/torchwave-public-eob-'+label+'.npz');assert 'dt_s' in z;dtm=float(z['dt_s'])/MSEC
  sha=hashlib.sha256(pathlib.Path('/tmp/torchwave-public-eob-'+label+'.npz').read_bytes()).hexdigest();b=np.asarray(z['mode22'],dtype=np.complex128);state=np.asarray(z['state'])
  tm=state[0]; epoch_s=float(z['epoch_s']); tb=np.arange(len(b))*dtm
  assert np.all(np.diff(tb)>0) and tm[0] <= tb[0] and tb[-1] >= tm[-1]
  assert tm[-1] > 4900.0
  valid=(tm>=tb[0])&(tm<=tb[-1]); tm_eval=tm[valid]
  b_on_state=interp_local(tb,b,tm_eval,5)
  expected_npz=json.load(open('docs/reviews/2026-09-13-parity-review/verification/eob/eob-clean-hamiltonian-public.json'))['results'][label]['vector_npz_sha256']
  assert sha == expected_npz
  om=SEOBNRv4PublishedFlux.from_hamiltonian(torch.from_numpy(state[1:].T.copy()),setup()[0])['omega'].numpy()
  base,_=evaluate(state[1:].T,tm,om,attach=False);bsi=b_on_state;bi=base[valid]*scale
  post=np.asarray(z['postpeak_state']); pt=post[0]; pst=post[1:].T; pom=SEOBNRv4PublishedFlux.from_hamiltonian(torch.from_numpy(pst.copy()),setup()[0])['omega'].numpy(); _,attached_public=evaluate(pst,pt,pom,attach=True)
  post_valid=(pt>=tb[0])&(pt<=tb[-1])&(pt>=4998.3293); post_binary=interp_local(tb,b,pt[post_valid],5)
  match_for_metrics=4998.3293
  results[label]={'npz_sha256':sha,'dt_M':dtm,'epoch_s':epoch_s,'mode_time_support_M':[float(tb[0]),float(tb[-1])],'state_time_support_M':[float(tm[0]),float(tm[-1])],'comparison_state_support_M':[float(tm_eval[0]),float(tm_eval[-1])],'carrier_on_identical_binary_state':metrics(bi,bsi,tm_eval),'clean_attachment_on_identical_binary_state':metrics(attached_public.mode.numpy()[post_valid]*scale,post_binary,pt[post_valid]),'orbital_phi_checkpoints':{str(x):float(state[2,min(int(np.searchsorted(tm,x)),len(tm)-1)]) for x in (0,100,1000,3000,4500,4900) if x<=tm[-1]}}
  _,patt=evaluate(pst,pt,pom)
  coeff=patt.calibration.coefficients
  ham,rr,_,_,_=setup(); stt=torch.from_numpy(state[1:].T.copy())
  with torch.inference_mode():
   nq=nqc_22_factor(stt[:,0],stt[:,2],torch.from_numpy(om),coeff,sample_dims=0).numpy()
  nqc_direct=base*nq*scale
  results[label]['nqc_attached_with_vector2_coefficients_on_vector1']=metrics(nqc_direct[valid],bsi,tm_eval)
  # Compare both exact-state analytic and seven-point published calibrations.
  analytic_report=json.load(open('docs/reviews/2026-09-13-parity-review/verification/eob/nqc-analytic-clean.json'))
  ar=analytic_report['results'][label]
  for cname in ('analytic','sampled'):
   coeff_a=torch.tensor(ar[cname+'_amplitude'],dtype=torch.float64); coeff_b=torch.tensor(ar[cname+'_phase'],dtype=torch.float64)
   coeff_obj=type(coeff)(coeff_a,coeff_b)
   with torch.inference_mode():
    factor=nqc_22_factor(stt[:,0],stt[:,2],torch.from_numpy(om),coeff_obj,sample_dims=0).numpy()
   wave=base*factor*scale
   match=float(ar['match_time_M'])
   pre=valid & (tm < match-1.0); late=valid & (tm >= match)
   results[label]['nqc_'+cname+'_coefficients']={'amplitude':coeff_a.tolist(),'phase':coeff_b.tolist(),'match_time_M':match,'prematch_lt_match_minus_1':metrics(wave[pre],bsi[pre],tm[pre]) if pre.any() else None,'merger_ge_match':metrics(wave[late],bsi[late],tm[late]) if late.any() else None}
  results[label]['public_vector2_nqc_coefficients']={'amplitude':coeff.amplitude.tolist(),'phase':coeff.phase.tolist(),'time_support_M':[float(pt[0]),float(pt[-1])]}
 results['saved_native_reference']={'native_attached_vs_public_not_recomputed':'saved native mode retained for reference','nqc_coefficients':{'amplitude':att_native.calibration.coefficients.amplitude.tolist(),'phase':att_native.calibration.coefficients.phase.tolist()}}
 report={'provenance':'Clean factorized 22 and NQC attachment evaluated independently on each trajectory state. Public binary mode is compared on its own time grid; no fitting or shifts. SI scale=(50 M_sun seconds)c/(100 Mpc). Public vector1/vector2 auxiliary rows are observed (t,r,phi,prstar,p_phi) schema. Tail phase is internal to the clean factorized helper and not separately exposed; orbital phi and total carrier phase are recorded.','saved_sha256':sh,'python':'/Users/xangma/miniconda3/envs/pycbcorig313/bin/python','results':results,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
