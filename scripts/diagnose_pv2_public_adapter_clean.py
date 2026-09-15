"""Public-to-public Pv2 source-frame adapter probe (no clean implementation)."""
import hashlib, json, pathlib
import numpy as np
import lal, lalsimulation as ls

CASES = [
    dict(label='frozen_catalog', m1=35.,m2=25.,s1=(.2,.1,.2),s2=(-.1,.15,-.1),inc=.7,ph=.3),
    dict(label='frozen_holdout', m1=42.,m2=17.,s1=(.3,-.15,-.3),s2=(.12,.08,.2),inc=1.1,ph=.71,fref=35.),
    dict(label='independent_low_mass', m1=20.,m2=15.,s1=(.1,.1,.2),s2=(0.,0.,0.),inc=.2,ph=0.),
    dict(label='independent_two_spin_a', m1=20.,m2=10.,s1=(.35,-.15,.1),s2=(.1,.2,-.25),inc=1.1,ph=-.2),
    dict(label='independent_two_spin_b', m1=12.,m2=11.,s1=(-.2,.25,-.1),s2=(.3,.05,.2),inc=.4,ph=.8),
]

def main():
    df=1.; f=np.arange(20.,513.,df); fv=lal.CreateREAL8Vector(len(f)); fv.data[:]=f
    rows=[]
    for c in CASES:
        rf=c.get('fref',20.)
        a=(c['m1']*lal.MSUN_SI,c['m2']*lal.MSUN_SI,rf,c['ph'],c['inc'],*c['s1'],*c['s2'])
        g=ls.SimIMRPhenomPCalculateModelParametersFromSourceFrame(*a,ls.IMRPhenomPv2_V)
        seq0=ls.SimIMRPhenomPFrequencySequence(fv,*g[:4],c['m1']*lal.MSUN_SI,c['m2']*lal.MSUN_SI,500e6*lal.PC_SI,g[4],g[5],rf,ls.IMRPhenomPv2_V,ls.NoNRT_V,lal.CreateDict())
        choose=ls.SimInspiralChooseFDWaveform(c['m1']*lal.MSUN_SI,c['m2']*lal.MSUN_SI,*c['s1'],*c['s2'],500e6*lal.PC_SI,c['inc'],c['ph'],0,0,0,df,20.,512.,rf,lal.CreateDict(),ls.IMRPhenomPv2)
        refp=np.asarray(choose[0].data.data)[20:].copy(); refc=np.asarray(choose[1].data.data)[20:].copy()
        ref=refp+1j*refc
        shp0=np.asarray(seq0[0].data.data).copy(); shc0=np.asarray(seq0[1].data.data).copy()
        intrinsic=shp0+1j*shc0
        # Public phase response: intrinsic and full APIs, independently.
        seqp=ls.SimIMRPhenomPFrequencySequence(fv,*g[:4],c['m1']*lal.MSUN_SI,c['m2']*lal.MSUN_SI,500e6*lal.PC_SI,g[4],g[5]+.1,rf,ls.IMRPhenomPv2_V,ls.NoNRT_V,lal.CreateDict())
        intrp=np.asarray(seqp[0].data.data).copy()+1j*np.asarray(seqp[1].data.data).copy()
        fullp=ls.SimInspiralChooseFDWaveform(c['m1']*lal.MSUN_SI,c['m2']*lal.MSUN_SI,*c['s1'],*c['s2'],500e6*lal.PC_SI,c['inc'],c['ph']+.1,0,0,0,df,20.,512.,rf,lal.CreateDict(),ls.IMRPhenomPv2)
        fullp=np.asarray(fullp[0].data.data)[20:].copy()+1j*np.asarray(fullp[1].data.data)[20:].copy()
        # Uniform public wrapper at the identical physical settings and
        # selected published coordinates (phi_aligned, -2*zeta).
        uni=ls.SimIMRPhenomP(*g[:4],c['m1']*lal.MSUN_SI,c['m2']*lal.MSUN_SI,500e6*lal.PC_SI,g[4],g[5],df,20.,512.,rf,ls.IMRPhenomPv2_V,ls.NoNRT_V,lal.CreateDict())
        uhp=np.asarray(uni[0].data.data)[20:].copy(); uhc=np.asarray(uni[1].data.data)[20:].copy()
        cz,sz=np.cos(2*g[6]),np.sin(2*g[6])
        # Explicit published spin-2 basis rotation: hp'=c hp+s hc,
        # hc'=c hc-s hp, equivalent to H'=exp(-2 i zeta)H.
        urhp,urhc=cz*uhp+sz*uhc,cz*uhc-sz*uhp
        sq=seq0
        shp,shc=np.asarray(sq[0].data.data).copy(),np.asarray(sq[1].data.data).copy()
        srhp,srhc=cz*shp+sz*shc,cz*shc-sz*shp
        def polmetrics(x,y): return {'full':err(x,y),'interior':err(x[:-1],y[:-1]),'endpoint_abs':float(abs(x[-1])),'reference_endpoint_abs':float(abs(y[-1]))}
        def circ(x): return float(np.angle(np.vdot(ref,x)))
        # ChooseFD's uniform wrapper may set its final f_high sample to zero;
        # retain that endpoint in the report while also exposing the common
        # interior support for a meaningful adapter comparison.
        def err(x,y): return float(np.linalg.norm(x-y)/np.linalg.norm(y))
        candidates=[]
        for phase_name,pc in [('plus_aligned',g[5]),('minus_aligned',-g[5]),('plus_aligned_plus_ref',g[5]+c['ph']),('minus_aligned_plus_ref',-g[5]+c['ph']),('plus_aligned_minus_ref',g[5]-c['ph']),('minus_aligned_minus_ref',-g[5]-c['ph'])]:
            # Recompute only the public intrinsic carrier phase coordinate.
            q=ls.SimIMRPhenomPFrequencySequence(fv,*g[:4],c['m1']*lal.MSUN_SI,c['m2']*lal.MSUN_SI,500e6*lal.PC_SI,g[4],pc,rf,ls.IMRPhenomPv2_V,ls.NoNRT_V,lal.CreateDict())
            qhp=np.asarray(q[0].data.data).copy(); qhc=np.asarray(q[1].data.data).copy()
            z=qhp+1j*qhc
            for rs in (-1,1):
                rot=z*np.exp(1j*rs*2*g[6])
                candidates.append({'phase':phase_name,'rotation_sign':rs,'relative_l2':err(rot,ref),'interior_relative_l2':err(rot[:-1],ref[:-1]),'endpoint_abs':float(abs(rot[-1])),'choose_endpoint_abs':float(abs(ref[-1])),'circular_phase_rad':circ(rot),'amplitude_relative_l2':float(np.linalg.norm(np.abs(rot)-np.abs(ref))/np.linalg.norm(np.abs(ref)))})
        selected=next(x for x in candidates if x['phase']=='plus_aligned' and x['rotation_sign']==-1)
        selected_hp=polmetrics(srhp,refp); selected_hc=polmetrics(srhc,refc)
        assert selected_hp['interior'] < 1e-12 and selected_hc['interior'] < 1e-12
        phase_delta=float(np.angle(np.vdot(intrinsic,intrp)))
        assert abs(np.angle(np.exp(1j*(phase_delta-.2)))) < 1e-12
        choose_meta={'length':len(choose[0].data.data),'f0':choose[0].f0,'deltaF':choose[0].deltaF,'epoch':str(choose[0].epoch)}
        seq_meta={'length':len(seq0[0].data.data),'f0':seq0[0].f0,'deltaF':seq0[0].deltaF,'epoch':str(seq0[0].epoch)}
        uniform_meta={'length':len(uni[0].data.data),'f0':uni[0].f0,'deltaF':uni[0].deltaF,'epoch':str(uni[0].epoch)}
        rows.append({'label':c['label'],'inputs':c,'sourceframe_phaseRef':c['ph'],'geometry':g,'sampling':{'sequence':seq_meta,'choose':choose_meta,'uniform':uniform_meta,'choose_slice':'indices20:','endpoint_note':'ChooseFD and uniform endpoints are retained; interior excludes only endpoint'},'phase_response':{'intrinsic_delta_.1_rad':phase_delta,'full_delta_.1_rad':float(np.angle(np.vdot(ref,fullp)))},'selected_adapter':selected,'selected_adapter_separate_pass':True,'polarizations':{'choose_vs_rotated_sequence':{'hp':selected_hp,'hc':selected_hc},'choose_vs_rotated_uniform':{'hp':polmetrics(urhp,refp),'hc':polmetrics(urhc,refc)},'uniform_vs_sequence':{'hp':polmetrics(urhp,srhp),'hc':polmetrics(urhc,srhc)}},'candidates':candidates})
    out=pathlib.Path('docs/reviews/2026-09-13-parity-review/pv2-clean');out.mkdir(parents=True,exist_ok=True)
    report={'provenance':'public CalculateModelParametersFromSourceFrame, FrequencySequence, SimIMRPhenomP, and ChooseFDWaveform APIs; discrete published phase/polarization choices only. Uniform output does not receive zeta; rotation is applied explicitly here.','api_docstrings':{'CalculateModelParametersFromSourceFrame':ls.SimIMRPhenomPCalculateModelParametersFromSourceFrame.__doc__,'FrequencySequence':ls.SimIMRPhenomPFrequencySequence.__doc__,'SimIMRPhenomP':ls.SimIMRPhenomP.__doc__,'ChooseFDWaveform':ls.SimInspiralChooseFDWaveform.__doc__},'cases':rows,'versions':{'lal':lal.__version__,'lalsimulation':ls.__version__},'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    (out/'public-adapter.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
