#!/usr/bin/env python3
"""Bounded raw-TD comparison of clean native TEOB output and public API."""
from pathlib import Path
import hashlib, json
import numpy as np
import lal
import lalsimulation as lalsim
import torch
from torchwave.approximants.eob.teob_conditioned_waveform_clean import condition_td_to_fd

ROOT=Path(__file__).resolve().parents[1]
VER=ROOT/'docs/reviews/2026-09-13-parity-review/verification/teob-conditioned-waveform-clean'
OUT=ROOT/'docs/reviews/2026-09-13-parity-review/verification/teob-first-difference-clean'
NATIVE=VER/'native-output.npz'
PRODUCER=ROOT/'src/torchwave/approximants/eob/teob_conditioned_waveform_clean.py'
CONDITIONER=ROOT/'src/torchwave/approximants/eob/td_fd_conditioning_clean.py'
def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 n=np.load(NATIVE); m1,m2=30.,20.; dt=float(n['sampling_dt_seconds']); fmin=float(n['support_start_frequency'])
 d=lal.CreateDict(); lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(d,0)
 ma=lalsim.SimInspiralCreateModeArray(); lalsim.SimInspiralModeArrayActivateMode(ma,2,2); lalsim.SimInspiralWaveformParamsInsertModeArray(d,ma)
 hp,hc=lalsim.SimIMRTEOBResumS(0.,dt,m1*lal.MSUN_SI,m2*lal.MSUN_SI,0.,0.,0.,0.,0.,0.,0.,0.,100e6*lal.PC_SI,0.,0.,d,0.,0.,fmin,0.)
 ph=np.asarray(hp.data.data,dtype=float); pc=np.asarray(hc.data.data,dtype=float)
 OUT.mkdir(parents=True,exist_ok=True); pub=OUT/'public-td-output.npz'
 np.savez(pub,hp=ph,hc=pc,epoch=float(hp.epoch),delta_t=float(hp.deltaT),f_min=fmin,m1=m1,m2=m2,distance_mpc=100.,inclination=0.,mode_l=2,mode_m=2)
 native=n['h_plus_td']+1j*n['h_cross_td']; public=-(ph+1j*pc); count=min(native.size,public.size); diff=native[:count]-public[:count]
 scale=max(float(np.max(np.abs(native))),np.finfo(float).tiny); threshold=max(1e-15*scale,1e-30)
 inds=np.flatnonzero(np.abs(diff)>threshold); first=int(inds[0]) if inds.size else None
 fd=condition_td_to_fd(torch.tensor(-ph,dtype=torch.float64),torch.tensor(-pc,dtype=torch.float64),dt=dt,delta_f=float(n['sampling_delta_f']),f_max=float(n['sampling_f_nyquist']),mass1=m1,mass2=m2,f_low=20.0,tc=0.,return_metadata=True)
 zfd=np.load(VER/'public-fd-sim-inspiral-output.npz'); pf=zfd['30p20_public_frequencies']; php=zfd['30p20_public_h_plus_fd']; phc=zfd['30p20_public_h_cross_fd']
 ff=fd.frequencies.detach().cpu().numpy(); fhp=fd.h_plus.detach().cpu().numpy(); fhc=fd.h_cross.detach().cpu().numpy(); assert np.array_equal(ff,pf); ii=np.flatnonzero((ff>=20.)&(ff<=1024.)); dp=fhp[ii]-php[ii]; dc=fhc[ii]-phc[ii]; fd_rel=float(np.linalg.norm(np.concatenate([dp,dc]))/max(np.linalg.norm(np.concatenate([php[ii],phc[ii]])),np.finfo(float).tiny))
 conditioned_file=OUT/'conditioned-public-td-fd.npz'; np.savez(conditioned_file,frequencies=ff,h_plus=fhp,h_cross=fhc)
 report={'status':'complete','model_identity':{'native':'clean TEOB conditioned waveform cache','public':'lalsimulation.SimIMRTEOBResumS','conditioner':'torchwave.approximants.eob.teob_conditioned_waveform_clean.condition_td_to_fd','lalsimulation':lalsim.__version__,'lal':lal.__version__,'producer_module_sha256':sha256(PRODUCER),'diagnostic_script_sha256':sha256(Path(__file__)),'conditioning_module_sha256':sha256(CONDITIONER)},'inputs':{'native_file':str(NATIVE),'native_sha256':sha256(NATIVE),'public_file':str(pub),'public_sha256':sha256(pub),'public_fd_cache':str(VER/'public-fd-sim-inspiral-output.npz'),'public_fd_cache_sha256':sha256(VER/'public-fd-sim-inspiral-output.npz')},'public_call':{'m1_msun':m1,'m2_msun':m2,'distance_mpc':100.,'inclination':0.,'phi_ref':0.,'delta_t_seconds':dt,'f_min_hz':fmin,'mode_array':[[2,2]],'PN_tidal_order':0,'f_ref_hz':0.,'spin1x':0.,'spin1y':0.,'spin1z':0.,'spin2x':0.,'spin2y':0.,'spin2z':0.,'lambda1':0.,'lambda2':0.,'eccentricity':0.,'mean_periastron_anomaly':0.,'long_asc_node':0.},'grid':{'native_samples':int(native.size),'public_samples':int(public.size),'native_dt_seconds':dt,'public_dt_seconds':float(hp.deltaT),'native_epoch_seconds':float(n['physical_epoch_seconds']),'public_epoch_seconds':float(hp.epoch),'epoch_difference_seconds':float(hp.epoch)-float(n['physical_epoch_seconds']),'comparison':'equal elapsed-from-generation-start indices; epochs differ; no physical-time alignment'},'raw_td_comparison':{'sign_contract':'-1 is project_teob22_direct versus generic/audited observer-frame convention in teob_conditioned_waveform_clean.py','shared_samples':count,'relative_l2_public_norm':float(np.linalg.norm(diff)/max(np.linalg.norm(public),np.finfo(float).tiny)),'max_abs':float(np.max(np.abs(diff))),'first_difference_index':first},'conditioned_public_td':{'status':'complete','fd_samples':int(fd.frequencies.numel()),'delta_f_hz':float(n['sampling_delta_f']),'f_max_hz':float(n['sampling_f_nyquist']),'fd_output_path':str(conditioned_file),'fd_output_sha256':sha256(conditioned_file)},'cached_public_fd_comparison':{'status':'complete','public_frequency_bins':int(pf.size),'frequency_range_hz':[20.0,1024.0],'requested_f_low_hz':20.0,'exact_bins':int(ii.size),'relative_l2_public_norm':fd_rel,'plus_relative_l2_public_norm':float(np.linalg.norm(dp)/max(np.linalg.norm(php[ii]),np.finfo(float).tiny)),'cross_relative_l2_public_norm':float(np.linalg.norm(dc)/max(np.linalg.norm(phc[ii]),np.finfo(float).tiny)),'max_abs':float(np.max(np.abs(np.concatenate([dp,dc]))))},'conclusion':'Raw TD polarizations differ at equal elapsed indices. Conditioning the sign-converted public TD also leaves a measured residual against the cached public SimInspiralFD output. These measurements alone do not separate generation inputs, waveform generation, and conditioner behavior; the independent public-stage replay is needed for attribution.'}
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
