"""Inspect one clean SEOB fine-grid replay without changing runtime code."""
from pathlib import Path
import hashlib, json, numpy as np, torch
from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
ROOT=Path(__file__).resolve().parents[1]; REPLAY=Path('/tmp/torchwave-seob-fine-replay')
def h(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def met(a,b):
 a=np.asarray(a); b=np.asarray(b); return {'n':int(a.size),'complex_rel_l2':float(np.linalg.norm(a-b)/np.linalg.norm(b)),'amp_rel_l2':float(np.linalg.norm(abs(a)-abs(b))/np.linalg.norm(abs(b))),'phase_rms':float(np.sqrt(np.mean(np.angle(a*b.conj())**2)))}
def main():
 with np.load(REPLAY/'native-full20-controller-clean.npz') as d: ft=d['fine_output_time']; fs=d['fine_output_state']; native=d['mode_100mpc']; tt=d['time']; valid=d['trajectory_valid']
 with np.load('/tmp/torchwave-public-eob-dt_0p5M.npz') as d: public=d['mode22']; dt=float(d['dt_s'])/(50*4.925490947641267e-6)
 n=min(native.size,public.size); full=np.zeros(public.size,complex); full[:n]=native[:n]
 nu=torch.tensor(.24,dtype=torch.float64); z=torch.tensor(0.,dtype=torch.float64); ham=SEOBNRv4Hamiltonian(nu,z,z,delta=torch.tensor(.2,dtype=torch.float64)); flux=SEOBNRv4PublishedFlux(spin_truncation='generic-leading-ell8')
 base,_=published_mode22_and_frequency(torch.from_numpy(fs),ham,flux,spin_v6='bohe_total_energy',include_pan_v8=False,include_positive_chi_540=False)
 # Fine local output begins at the documented restart time 4853M; compare exact .5M knots only.
 restart=4853.; pick=(np.rint((ft+restart)/.5)*.5 == ft+restart); pi=np.rint((ft[pick]+restart)/.5).astype(int); good=pi<public.size
 scale=50*4.925490947641267e-6*299792458/(100*3.0856775814913673e22)
 report={'replay':{'directory':str(REPLAY),'startup_sha256':h(REPLAY/'startup.json'),'diagnostic_sha256':h(REPLAY/'diagnostic.json'),'cache_sha256':h(REPLAY/'native-full20-controller-clean.npz')},'inputs':{'public_sha256':h('/tmp/torchwave-public-eob-dt_0p5M.npz'),'runtime_script_sha256':h(ROOT/'scripts/diagnose_eob_full20_controller_clean.py'),'fine_stage_script_sha256':h(__file__)},'events':json.loads((REPLAY/'diagnostic.json').read_text())['adaptive'],'comparisons':{'cached_native_full_padded':met(full,public),'fine_base_exact_shared_timestamps':met(base.detach().numpy()[pick][good]*scale,public[pi[good]]),'fine_base_sample_count':int(good.sum())},'limitations':['Replay cache stores only assembled mode and fine state, not fine base/NQC corrected complex modes. The fine base comparison uses exact shared .5M timestamps after the 4853M restart and cannot attribute the residual to source versus attachment. No numerical fix is supported by these observables.']}
 out=ROOT/'docs/reviews/2026-09-13-parity-review/verification/eob-fine-stage-replay-clean/report.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
