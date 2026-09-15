"""Black-box NSBH amplitude-component diagnostic; no production correction."""
import json, hashlib, inspect, math
from pathlib import Path
import torch
import torchwave.approximants.imrphenom_nsbh as n
from parity_harness import get_canonical_catalog_cases,get_canonical_holdout_cases,run_oracle_comparison

PYTHON='/Users/xangma/miniconda3/envs/pycbcorig313/bin/python'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def win(x,x0,d,sign): return .5*(1+sign*math.tanh(4*(float(x)-x0)/d))
def paper(p):
 m1,m2,s,l,_=p; M=m1+m2; eta=m1*m2/M**2; q=m1/m2
 t=torch.tensor
 tt=[t(v,dtype=torch.float64) for v in (m1,m2,s,l)]
 ft,fi,fr,c=n.nsbh_frequencies(*tt)
 frt=(.99*.98*fr if l>1 else (1-.02*l+.01*l*l)*.98*fr)
 mf,chi=n.zappa_remnant_properties(tt[0],tt[1],tt[2],t(0.,dtype=torch.float64),tt[3])
 _,qrd=n.london_ringdown_properties(mf,chi)
 xi,_=n.solve_xi_tide(tt[1]/tt[0],tt[2],c); ri=n.kerr_isco_radius(tt[2])
 tor=.296*xi*(1-2*c)-.171*q*c*ri
 tor=max(float(tor),0.)
 xnd=((ft-frt)/frt)**2-.571505*c-.00508451*s
 xndp=((ft-frt)/frt)**2-.657424*c-.0259977*s
 xd=tor+.424912*c+.363604*eta**.5-.060559*s
 xd_2015=tor+.424912*c+.363604*eta**.5-.0605591*s
 xdp=tor-.132754*c+.576669*eta**.5-.0603749*s-.0601185*s*s-.0729134*s**3
 eps=1.0 if float(ft)>=float(fr) else (1.29971-1.61724*xd)
 epsilon_tide=float(win(xnd,-.0796251,.0801192,1)) if float(ft)>=float(fr) else 0.0
 sigma_tide=float(win(xndp,-.206465,.226844,-1)) if float(ft)>=float(fr) else float((win(xndp,-.206465,.226844,-1)+.137722-.293237*xdp)/2)
 f0=frt if float(ft)>=float(fr) else ((q-1)*frt+eps*ft)/q
 f1=frt if float(ft)>=float(fr) else ((q-1)*frt+ft)/q
 if l>1:
  gp=1.25; d2p=1.62496/2*win((ft-frt)/frt,.0188092,.338737,1)
 else: gp=float(1+.5*l-.25*l*l); d2p=None
 try:
  import lalsimulation as ls, lal
  d2_public=float(ls.SimIMRPhenomNSBH_delta2_prime(float(ft),float(frt))) if l>1 else None
  props=[float(x) for x in ls.SimIMRPhenomNSBHProperties(m1*lal.MSUN_SI,m2*lal.MSUN_SI,s,l)]
 except ImportError:
  d2_public=None
  props=None
 return dict(m1=m1,m2=m2,compactness=float(c),f_tide=float(ft),f_rd=float(fr),f_rd_tilde=float(frt),ringdown_q=float(qrd),torus_ratio=tor,x_nd=float(xnd),x_nd_prime=float(xndp),x_d=float(xd),x_d_2015=float(xd_2015),x_d_prime=float(xdp),epsilon_ins=eps,epsilon_tide=epsilon_tide,sigma_tide=sigma_tide,window_width=.015+sigma_tide,f0=float(f0),f1=float(f1),f2=float(frt),gamma1_prime=gp,gamma1_prime_printed=float(1-.5*l-.25*l*l),delta2_prime_formula=None if d2p is None else float(d2p),delta2_prime_public=d2_public,public_properties=props)

def original2015(p, qnew):
 """Paper 1509.00512 Sec. IV prescriptions, retained for comparison only."""
 q=p[0]/p[1]; fr=qnew['f_rd']; ft=qnew['f_tide']; ftilde=qnew['f_rd_tilde']; eps=qnew['epsilon_ins']
 # Eq. (18) has the pre-2020 factor of two; Eq. (24) is used directly for
 # mildly disruptive no-torus systems.  The Q-weighted f1/f2 are Eq. (26).
 sigma_nd=2.*float(win(qnew['x_nd_prime'],-.206465,.226844,-1))
 sigma_mild=.137722-.293237*qnew['x_d_prime']
 nondisruptive=ft>=qnew['f_rd']
 d2_original=None if p[3] <= 1 else 1.62496*float(win((ft-ftilde)/ftilde,.0188092,.338737,-1))
 if nondisruptive:
  return {'m1':qnew['m1'],'m2':qnew['m2'],'f_rd':qnew['f_rd'],'ringdown_q':qnew['ringdown_q'],'source':'arXiv:1509.00512 Sec IV.0.3: epsilon Eq18, ND sigma Eq20, delta2 Eq22','branch':'nondisruptive','epsilon_tide':2.*float(win(qnew['x_nd'],-.0796251,.0801192,1)),'sigma_tide':sigma_nd,'delta2_prime':d2_original,'f0':ftilde,'f1':ftilde,'f2':ftilde,
          'large_ft_limit':{'x_values':[0.,1.,10.],'delta2_prime':[None if d2_original is None else 1.62496*win(x,.0188092,.338737,-1) for x in (0.,1.,10.)],
                             'interpretation':'printed omega_minus tends to zero as f_tide/f_RD_tilde grows; no sign alternative applied'}}
 # In the original paper Q denotes the binary mass ratio (line preceding
 # Eq. 21), so these are algebraically the current q-weighted centers.
 sigma_mild=(sigma_nd+sigma_mild)/2.
 return {'m1':qnew['m1'],'m2':qnew['m2'],'f_rd':qnew['f_rd'],'ringdown_q':qnew['ringdown_q'],'source':'arXiv:1509.00512 Sec IV.0.3: mild sigma average Eq28; epsilon Eq18, ND sigma Eq20, delta2 Eq22','branch':'mild_no_torus','epsilon_tide':0.,'sigma_tide':sigma_mild,'delta2_prime':d2_original,'f0':((q-1)*ftilde+eps*ft)/q,'f1':((q-1)*ftilde+ft)/q,'f2':ftilde,'Q_mass_ratio':q,
         'large_ft_limit':{'x_values':[0.,1.,10.],'delta2_prime':[None if d2_original is None else 1.62496*win(x,.0188092,.338737,-1) for x in (0.,1.,10.)],
                            'interpretation':'printed omega_minus tends to zero as f_tide/f_RD_tilde grows; no sign alternative applied'}}
SAMPLE_HZ=[30.,100.,500.,1000.,1500.,2500.]
def run(p, mode=None, unwindowed=False, frequencies=None):
 f=torch.tensor(SAMPLE_HZ if frequencies is None else frequencies,dtype=torch.float64); calls=[]; orig=n._amplitude_window
 def hook(freq,center,width,m,*,rising):
  i=len(calls); actual=orig(freq,center,width,m,rising=rising)
  calls.append(dict(center=float(center),width=float(width),rising=bool(rising),values=[float(x) for x in actual]))
  if mode is None or mode==i: return (torch.ones_like(freq) if unwindowed else actual)
  return torch.zeros_like(freq)
 n._amplitude_window=hook
 try:
  a=[torch.tensor(v,dtype=torch.float64) for v in p[:3]]
  hp,hc=n.IMRPhenomNSBH()(f,*a,lambda2=torch.tensor(p[3],dtype=torch.float64),f_ref=p[4],inclination=.5,phic=.3)
 finally: n._amplitude_window=orig
 return calls,dict(hp_abs=[float(x) for x in hp.abs()],hc_abs=[float(x) for x in hc.abs()],hp_real=[float(x) for x in hp.real],hp_imag=[float(x) for x in hp.imag],hc_real=[float(x) for x in hc.real],hc_imag=[float(x) for x in hc.imag])

def binary(p, frequencies=None):
 """Public LAL sequence, sampled at the same six frequencies as run()."""
 try:
  import lal, lalsimulation as ls
 except ImportError:
  return {'status':'SKIPPED','reason':'LALSimulation not installed'}
 m1,m2,s,l,fref=p; extra=lal.CreateDict()
 ls.SimInspiralWaveformParamsInsertTidalLambda1(extra,0.)
 ls.SimInspiralWaveformParamsInsertTidalLambda2(extra,float(l))
 ap=ls.GetApproximantFromString('IMRPhenomNSBH'); df=.25; fhi=4096.
 hp,hc=ls.SimInspiralChooseFDWaveform(float(m1)*lal.MSUN_SI,float(m2)*lal.MSUN_SI,0.,0.,float(s),0.,0.,0.,100.*1e6*lal.PC_SI,.5,.3,0.,0.,0.,df,30.,fhi,float(fref),extra,ap)
 sample=list(SAMPLE_HZ if frequencies is None else frequencies)
 idx=[round(x/df) for x in sample]
 return {'status':'OK','frequency_hz':sample, 'hp_real':[float(hp.data.data[i].real) for i in idx], 'hp_imag':[float(hp.data.data[i].imag) for i in idx], 'hc_real':[float(hc.data.data[i].real) for i in idx], 'hc_imag':[float(hc.data.data[i].imag) for i in idx]}

def sequence_control(p):
 try:
  import lal, lalsimulation as ls
 except ImportError:
  return {'status':'SKIPPED'}
 v=lal.CreateREAL8Vector(len(SAMPLE_HZ)); v.data[:]=SAMPLE_HZ
 e=lal.CreateDict(); ls.SimInspiralWaveformParamsInsertTidalLambda1(e,0.); ls.SimInspiralWaveformParamsInsertTidalLambda2(e,float(p[3]))
 s=ls.SimIMRPhenomNSBHFrequencySequence(v,.3,float(p[4]),100e6*lal.PC_SI,float(p[0])*lal.MSUN_SI,float(p[1])*lal.MSUN_SI,float(p[2]),0.,e)
 e2=lal.CreateDict(); ls.SimInspiralWaveformParamsInsertTidalLambda1(e2,0.); ls.SimInspiralWaveformParamsInsertTidalLambda2(e2,float(p[3]))
 h=ls.SimInspiralChooseFDWaveform(float(p[0])*lal.MSUN_SI,float(p[1])*lal.MSUN_SI,0.,0.,float(p[2]),0.,0.,0.,100e6*lal.PC_SI,0.,.3,0.,0.,0.,.25,30.,4096.,float(p[4]),e2,ls.GetApproximantFromString('IMRPhenomNSBH'))[0]
 vals=[]
 for i,f in enumerate(SAMPLE_HZ):
  y=h.data.data[round(f/.25)]; x=s.data.data[i]; vals.append({'f_hz':f,'relative_complex':abs(x-y)/(abs(y) or 1.),'sequence_abs':abs(x),'choosefd_abs':abs(y)})
 return {'status':'OK','face_on':True,'values':vals}

def checks(z, lkey):
 a=z[lkey]['unwindowed_window0']; b=z[lkey]['unwindowed_window1']
 # Window-isolated outputs are also retained; these checks are explicit
 # component controls, rather than target comparisons.
 return {'window_vectors':z[lkey]['windows'],
         'pn_finite':all(math.isfinite(x) for x in a['hp_abs']+a['hc_abs']),
         'premerger_finite':all(math.isfinite(x) for x in b['hp_abs']+b['hc_abs'])}

def binary_point_residual(native, ref):
 if ref.get('status') != 'OK': return {'status':ref.get('status'), 'reason':ref.get('reason')}
 out={}
 for pol in ('hp','hc'):
  vals=[]
  for i in range(6):
   x=complex(native[pol+'_real'][i],native[pol+'_imag'][i]); y=complex(ref[pol+'_real'][i],ref[pol+'_imag'][i])
   if abs(y) == 0.:
    vals.append({'relative_complex':None,'relative_magnitude':None,'zero_support':True,'support_mismatch':bool(abs(x)>0.)})
   else:
    vals.append({'relative_complex':abs(x-y)/abs(y),'relative_magnitude':abs(abs(x)-abs(y))/abs(y),'zero_support':False})
  out[pol]=vals
 return out

def rd_shape(p, component, q):
 """Eq. (27)-(28) shape, with normalization removed at 1000 Hz."""
 if q.get('delta2_prime_formula') is None or q.get('epsilon_tide',0.) == 0.: return {'status':'NOT_APPLICABLE'}
 t=torch.tensor
 m1,m2,s,l,_=p
 # Public callable returns f_RD and Q; Eq. (28) uses sigma=delta2' f_RD/Q.
 _,_,fr,c=n.nsbh_frequencies(t(m1,dtype=torch.float64),t(m2,dtype=torch.float64),t(s,dtype=torch.float64),t(l,dtype=torch.float64))
 mf,chi=n.zappa_remnant_properties(t(m1,dtype=torch.float64),t(m2,dtype=torch.float64),t(s,dtype=torch.float64),t(0.,dtype=torch.float64),t(l,dtype=torch.float64))
 _,qq=n.london_ringdown_properties(mf,chi); sigma=q['delta2_prime_formula']*float(fr)/float(qq)
 fs=SAMPLE_HZ; fn=1000.; den=lambda f:(f-float(fr))**2+(sigma*sigma/4.)
 expected=[(f/fn)**(-7./6.)*den(fn)/den(f) for f in fs]
 vals=component['hp_abs']; norm=vals[fs.index(1000.)]
 observed=[v/norm if norm else None for v in vals]
 return {'sigma_hz':sigma,'f_rd_hz':float(fr),'Q':float(qq),'normalization_hz':fn,'expected_ratio':expected,'observed_ratio':observed,'relative_error':[None if o is None else o-e for o,e in zip(observed,expected)]}

def reconstruction_error(total, parts):
 vals=[]
 for pol in ('hp','hc'):
  e=[]
  for i in range(len(total[pol+'_real'])):
   x=complex(total[pol+'_real'][i],total[pol+'_imag'][i])
   y=sum(complex(p[pol+'_real'][i],p[pol+'_imag'][i]) for p in parts)
   e.append(None if abs(x)==0. and abs(y)==0. else (None if abs(x)==0. else abs(x-y)/abs(x)))
  vals.append(e)
 return {'max_relative_hp':max((x for x in vals[0] if x is not None),default=0.),'max_relative_hc':max((x for x in vals[1] if x is not None),default=0.),'support_mismatch':any(x is None and abs(complex(total[pol+'_real'][i],total[pol+'_imag'][i]))>0 for pol in ('hp','hc') for i,x in enumerate(vals[0 if pol=='hp' else 1]))}

def original_candidate(raw, spec, current_spec):
 # _amplitude_window receives dimensionless Mf; convert both samples and
 # published Hz centers with the exact total-mass seconds factor.
 msec=(spec.get('m1',0.)+spec.get('m2',0.))*4.92549095e-6
 if not msec: msec=(current_spec.get('m1',0.)+current_spec.get('m2',0.))*4.92549095e-6
 f=torch.tensor(SAMPLE_HZ,dtype=torch.float64)*msec
 d=(.015+spec['sigma_tide'])
 ws=[.5*(1+(-1 if i<2 else 1)*torch.tanh(4*(f-spec[k]*msec)/d)) for i,k in enumerate(('f0','f1','f2'))]
 out={}
 rd_scale=torch.ones_like(f)
 if (spec.get('epsilon_tide') and spec.get('delta2_prime') is not None and
     current_spec.get('epsilon_tide') and current_spec.get('delta2_prime_formula')):
  sig_old=spec['delta2_prime']*float(spec['f_rd'])*msec/float(spec['ringdown_q'])
  sig_new=current_spec['delta2_prime_formula']*float(current_spec['f_rd'])*msec/float(current_spec['ringdown_q'])
  fr=float(current_spec['f_rd'])*msec
  rd_scale_eps=spec['epsilon_tide']/current_spec['epsilon_tide']
  rd_scale=(rd_scale_eps*(sig_old/sig_new)**2*((f-fr)**2+sig_new**2/4)/((f-fr)**2+sig_old**2/4))
 for pol in ('hp','hc'):
  z=[]
  for i in range(6):
   z.append(sum(complex(raw[j][pol+'_real'][i],raw[j][pol+'_imag'][i])*float(ws[j][i])*(float(rd_scale[i]) if j==2 else 1.) for j in range(3)))
  out[pol]=z
 return out

def candidate_residual(cand, ref):
 if ref.get('status')!='OK': return {'status':ref.get('status')}
 out={}
 for pol in ('hp','hc'):
  num=den=mag_num=0.; mags=[]
  for i,z in enumerate(cand[pol]):
   y=complex(ref[pol+'_real'][i],ref[pol+'_imag'][i]); num+=abs(z-y)**2; den+=abs(y)**2; mag_num+=(abs(z)-abs(y))**2
   mags.append(None if abs(y)==0 else abs(z)/abs(y))
  out[pol]={'relative_l2':math.sqrt(num/den) if den else None,'magnitude_relative_l2':math.sqrt(mag_num/den) if den else None,'magnitude_ratio':mags}
 return out

def lorentzian_controls(spec, current):
 if spec.get('delta2_prime') is None or current.get('delta2_prime_formula') is None or current.get('epsilon_tide',0.) == 0.:
  return {'status':'NOT_APPLICABLE'}
 msec=(spec['m1']+spec['m2'])*4.92549095e-6
 q=float(spec['ringdown_q']); fr=float(spec['f_rd'])*msec
 so=spec['delta2_prime']*fr/q; sn=current['delta2_prime_formula']*fr/q
 er=spec['epsilon_tide']/current['epsilon_tide']
 def ratio(f): return er*(so/sn)**2*((f-fr)**2+sn**2/4)/((f-fr)**2+so**2/4)
 return {'sigma_old_Mf':so,'sigma_current_Mf':sn,'epsilon_ratio':er,
         'at_f_rd':ratio(fr),'far_tail_f_hz_30':ratio(30.*msec),
         'far_tail_limit':er*(so/sn)**2}

def window_comparison(c, q):
 expected=[q['f0'],q['f1'],q['f2']]
 return [{'center_expected':expected[i],'center_captured':c[i]['center'],
          'center_error':c[i]['center']-expected[i],
          'width_expected':q['window_width'],'width_captured':c[i]['width'],
          'width_error':c[i]['width']-q['window_width'],
          'values_captured':c[i]['values']} for i in range(min(3,len(c)))]

def main():
 cases={'independent_c':(3.2,1.2,-.25,800.,32.),'independent_f':(9.,1.1,-.4,2000.,42.)}
 out={'metadata':{'python':PYTHON,'source_sha256':{x:sha(x) for x in ['scripts/nsbh_amplitude_components_diagnostic.py','src/torchwave/approximants/imrphenom_nsbh.py']},'paper':'arXiv:2002.08383 Appendix A Eqs 20-24,26,29-30'}}
 for name,p in cases.items():
  rec={'case':name,'paper':paper(p),'lambda0':{},'restored':{}}
  for lkey,l in [('lambda0',0.),('restored',p[3])]:
   pp=(*p[:3],l,p[4]);
   c,tot=run(pp); _,pn=run(pp,0); _,pm=run(pp,1); _,pr=run(pp,2)
   pq=paper(pp); rec[lkey]={'windows':c,'paper':pq,'original_2015':original2015(pp,pq),'window_comparison':window_comparison(c,pq),'total':tot,'components_window0':pn,'components_window1':pm,'components_window2':pr,'unwindowed_window0':run(pp,0,True)[1],'unwindowed_window1':run(pp,1,True)[1],'unwindowed_window2':run(pp,2,True)[1]}
   rec[lkey]['checks']=checks(rec,lkey)
   rec[lkey]['reconstruction']=reconstruction_error(tot,[pn,pm,pr])
   assert max(rec[lkey]['reconstruction'].values()) < 1e-10
  # Component identity controls: the lambda-dependent PN carrier must be
  # unchanged; record the observed premerger complex ratio explicitly.
  rec['component_identity']={}
  for comp in ('unwindowed_window0','unwindowed_window1','unwindowed_window2'):
   x=rec['lambda0'][comp]; y=rec['restored'][comp]; vals=[]
   for i in range(6):
    xa=complex(x['hp_real'][i],x['hp_imag'][i]); ya=complex(y['hp_real'][i],y['hp_imag'][i])
    vals.append({'hp_ratio_real':(ya/xa).real if xa else None,'hp_ratio_imag':(ya/xa).imag if xa else None,
                 'hp_magnitude_ratio':None if abs(xa)==0. and abs(ya)==0. else (None if abs(xa)==0. else abs(ya)/abs(xa))})
   rec['component_identity'][comp]=vals
  if name == 'independent_c':
   base=run((*p[:3],0.,p[4]),1,True)[1]
   scan=[]
   for lv in (0.,.25,.5,.75,1.,1.25):
    cur=run((*p[:3],lv,p[4]),1,True)[1]
    z=complex(cur['hp_real'][2],cur['hp_imag'][2]); b=complex(base['hp_real'][2],base['hp_imag'][2])
    pred=1.0+.5*lv-.25*lv*lv if lv<=1. else 1.25
    scan.append({'lambda2':lv,'observed_ratio':abs(z)/abs(b),'predicted_ratio':pred})
   rec['small_lambda_premerger_scan']=scan
   # Genuine public-oracle magnitude control; complex phases are excluded.
   oracle_scan=[]
   for label,basep in [('C',p),('positive_spin',(6.,1.4,.2,700.,35.))]:
    rows=[]
    for lv in (.25,.5,.75,1.,1.25):
     pp=(*basep[:3],lv,basep[4]); nt=run(pp)[1]; nb=binary(pp)
     rows.append({'lambda2':lv,'hp_magnitude_ratio':[abs(complex(nt['hp_real'][i],nt['hp_imag'][i]))/abs(complex(nb['hp_real'][i],nb['hp_imag'][i])) for i in range(6)],
                  'hc_magnitude_ratio':[abs(complex(nt['hc_real'][i],nt['hc_imag'][i]))/abs(complex(nb['hc_real'][i],nb['hc_imag'][i])) for i in range(6)]})
    oracle_scan.append({'case':label,'rows':rows})
   rec['public_oracle_magnitude_scan']=oracle_scan
   # Discrete composition hypotheses, named by physical window.  In the
   # diagnostic schema f0 is the PN window and f1 is the PM window; paper
   # 2015's f1 therefore maps to diagnostic f0 (paper 2020's f0).
   center_rows=[]
   for label,cp in [('C',p),('neighbor_a',(4.,1.3,-.2,1000.,35.)),('neighbor_b',(5.,1.4,-.1,1200.,40.))]:
    qv=paper(cp); row={'case':label,'branch':'mild_no_torus' if qv['f_tide']<qv['f_rd'] and qv['torus_ratio']==0. else 'other','epsilon_ins':qv['epsilon_ins'],'x_d_2015_minus_x_d':qv['x_d_2015']-qv['x_d']}
    if row['branch']=='mild_no_torus':
     pp=(*cp[:3],cp[3],cp[4]); rawv=[run(pp,i,True)[1] for i in range(3)]; refv=binary(pp)
     qvar=dict(qv); qvar['f1']=qv['epsilon_ins']*qv['f1']
     row['pm_base_f1_hz']=qv['f1']; row['pm_variant_f1_hz']=qvar['f1']
     row['pm_variant']=candidate_residual(original_candidate(rawv,qvar,qv),refv)
     # Paper 2015 f1 is the PN (first) window, corresponding to diagnostic
     # f0; retain the separate PM choice above as a named alternative.
     eps15=1.29971-1.61724*qv['x_d_2015']
     qpn=dict(qv); qpn['f0']=((cp[0]/cp[1]-1)*qv['f_rd_tilde']+eps15*qv['f_tide'])/(cp[0]/cp[1])
     row['pn_base_f0_hz']=qv['f0']; row['pn_coefficient_variant_f0_hz']=qpn['f0']; row['epsilon_ins_2015']=eps15
     row['pn_coefficient_variant']=candidate_residual(original_candidate(rawv,qpn,qv),refv)
     # Exact outer-PN composition hypothesis: scale the complete PN center.
     qpn_outer=dict(qv); qpn_outer['f0']=qv['epsilon_ins']*qv['f0']
     row['pn_outer_variant_f0_hz']=qpn_outer['f0']
     row['pn_outer_variant']=candidate_residual(original_candidate(rawv,qpn_outer,qv),refv)
     raw_pm=[dict(x) for x in rawv];
     for key in ('hp_real','hp_imag','hc_real','hc_imag'): raw_pm[2][key]=[0.]*len(raw_pm[2][key])
     row['pn_outer_premerger_only']=candidate_residual(original_candidate(raw_pm,qpn_outer,qv),refv)
    center_rows.append(row)
   rec['epsilon_center_variant']=center_rows
  rec['binary_independent']={'lambda0':binary((*p[:3],0.,p[4])),'restored':binary(p)}
  rec['frequency_sequence_control']=sequence_control(p)
  rec['binary_residual']={k:binary_point_residual(rec[k]['total'],rec['binary_independent'][k]) for k in ('lambda0','restored')}
  rec['current_identity']={}
  for lkey in ('lambda0','restored'):
   raw=[rec[lkey]['unwindowed_window'+str(i)] for i in range(3)]
   # Manufactured current prescription must reconstruct native components.
   cur=original_candidate(raw,rec[lkey]['paper'],rec[lkey]['paper'])
   rec['current_identity'][lkey]=candidate_residual(cur,{'status':'OK',**rec[lkey]['total']})
  rec['original_candidate']={}
  lkey='restored'
  raw=[rec[lkey]['unwindowed_window'+str(i)] for i in range(3)]
  rec['original_candidate'][lkey]=candidate_residual(original_candidate(raw,rec[lkey]['original_2015'],rec[lkey]['paper']),rec['binary_independent'][lkey])
  if rec[lkey]['paper'].get('delta2_prime_public') is not None:
   pub=dict(rec[lkey]['paper']); pub['delta2_prime']=pub['delta2_prime_public']; pub['epsilon_tide']=rec[lkey]['paper']['epsilon_tide']; pub['sigma_tide']=rec[lkey]['paper']['sigma_tide']
   rec['public_delta2_candidate']=candidate_residual(original_candidate(raw,pub,rec[lkey]['paper']),rec['binary_independent'][lkey])
  rec['lorentzian_controls']=lorentzian_controls(rec[lkey]['original_2015'],rec[lkey]['paper'])
  if name == 'independent_c':
   # Source-convention diagnostic: recompute only x'_ND using f_RD rather
   # than adjusted f_RD_tilde; all centers and other quantities stay fixed.
   qp=dict(rec[lkey]['paper'])
   xfr=((qp['f_tide']-qp['f_rd'])/qp['f_rd'])**2-.657424*qp['compactness']-.0259977*p[2]
   sig_nd_frd=win(xfr,-.206465,.226844,-1)
   qp['sigma_tide']=float(sig_nd_frd if qp['f_tide']>=qp['f_rd'] else (sig_nd_frd+.137722-.293237*qp['x_d_prime'])/2)
   qp['window_width']=.015+qp['sigma_tide']
   rec['c_width_frd_variant']={'x_nd_prime_frd':xfr,'sigma_tide':qp['sigma_tide'],
      'candidate':candidate_residual(original_candidate(raw,qp,rec[lkey]['paper']),rec['binary_independent'][lkey])}
  # Diagnostic-only complement of the printed large-ft omega-minus in Eq. 29.
  # It preserves A/2 and changes only omega-minus -> omega-plus; no runtime use.
  if name == 'independent_f':
   alt=dict(rec[lkey]['original_2015'])
   x=(rec[lkey]['paper']['f_tide']-rec[lkey]['paper']['f_rd_tilde'])/rec[lkey]['paper']['f_rd_tilde']
   alt['delta2_prime']=1.62496/2.*float(win(x,.0188092,.338737,1))
   alt['source'] += '; diagnostic omega-plus complement only'
   rec['complement_omega_plus_candidate']={'delta2_prime':alt['delta2_prime'],'x':x,
     'candidate':candidate_residual(original_candidate(raw,alt,rec[lkey]['paper']),rec['binary_independent'][lkey])}
  rec['ringdown_shape']=rd_shape(p,rec['restored']['unwindowed_window2'],rec['restored']['paper'])
  # Exact public sequence points: every frequency is an integer multiple of
  # df=.25 Hz, while retaining a 1-Hz dense report through 4096 Hz.
  dense_freq=[30.+.25*i for i in range(0,16265,4)]
  assert all(abs((f/.25)-round(f/.25)) < 1e-12 for f in dense_freq)
  rec['dense_amplitude_ratio']={}
  for lkey,l in [('lambda0',0.),('restored',p[3])]:
   pp=(*p[:3],l,p[4]); _,nt=run(pp,frequencies=dense_freq); nb=binary(pp,frequencies=dense_freq)
   def ratios(pol):
    vals=[]; support=[]
    for i in range(len(dense_freq)):
     x=abs(complex(nt[pol+'_real'][i],nt[pol+'_imag'][i])); y=abs(complex(nb[pol+'_real'][i],nb[pol+'_imag'][i])); ok=y>0.
     support.append(ok); vals.append(x/y if ok else None)
    return vals,support
   hpv,hps=ratios('hp'); hcv,hcs=ratios('hc')
   rec['dense_amplitude_ratio'][lkey]={'frequency_hz':dense_freq,'hp_native_over_binary':hpv,'hp_binary_nonzero':hps,'hc_native_over_binary':hcv,'hc_binary_nonzero':hcs}
  out.setdefault('records',[]).append(rec)
  # strict public binary summaries are emitted once below
  base=next(c for c in get_canonical_catalog_cases() if c.get('torch_name')=='IMRPhenomNSBH')
  if name == 'independent_f':
   for tag,case in [('catalog',base),('holdout',next(c for c in get_canonical_holdout_cases() if c.get('torch_name')=='IMRPhenomNSBH'))]:
    r=run_oracle_comparison(case,strict=False); out.setdefault('binary_controls',[]).append({'case':tag,'plus':r.get('plus'),'cross':r.get('cross')})
 Path('docs/nsbh_amplitude_components_diagnostic.json').write_text(json.dumps(out,indent=2,allow_nan=False,default=lambda x: float(x) if hasattr(x, 'item') else str(x)))
main()
