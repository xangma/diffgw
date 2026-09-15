"""Public aligned-limit Pv2/D time-origin probe; no runtime calibration."""
import hashlib, json, pathlib, math
import numpy as np, torch, lal, lalsimulation as ls
from scipy.interpolate import CubicSpline
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.kerr_qnm_generated import signed_qnm_220
torch.set_default_dtype(torch.float64)
CASES=[dict(label='A',m1=35.,m2=25.,z1=.2,z2=-.1),dict(label='B',m1=45.,m2=15.,z1=.3,z2=-.25),dict(label='independent',m1=27.,m2=16.,z1=-.25,z2=.15)]
F=np.arange(20.,513.,1.,dtype=np.float64)
def rel(a,b): return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))
def fs(f):
    v=lal.CreateREAL8Vector(len(f)); v.data[:]=np.asarray(f,dtype=np.float64); return v
def public_d(f,c):
    h=ls.SimIMRPhenomDFrequencySequence(fs(f),0.,20.,c['m1']*lal.MSUN_SI,c['m2']*lal.MSUN_SI,c['z1'],c['z2'],500e6*lal.PC_SI,lal.CreateDict(),ls.NoNRT_V)
    return np.asarray(h.data.data).copy()
def public_p(f,c,fref=20.):
    h=ls.SimIMRPhenomPFrequencySequence(fs(f),c['z1'],c['z2'],0.,0.,c['m1']*lal.MSUN_SI,c['m2']*lal.MSUN_SI,500e6*lal.PC_SI,0.,0.,fref,ls.IMRPhenomPv2_V,ls.NoNRT_V,lal.CreateDict())
    return np.asarray(h[0].data.data).copy(),np.asarray(h[1].data.data).copy()
def slope(c,frd,h):
    z=public_d(np.array([frd-h,frd+h]),c); return float(np.angle(np.conj(z[0])*z[1])/(2*h))
def slope5(c,frd,h):
    z=public_d(frd+np.array([-2*h,-h,h,2*h]),c)
    dm=np.angle(np.conj(z[1])*z[2])/(2*h); outer=np.angle(np.conj(z[0])*z[3])/(4*h)
    return float((4*dm-outer)/3)
def spline_slope(c,frd,lo,hi,n,bc):
    x=np.linspace(lo*frd,hi*frd,n); z=public_d(x,c); ph=np.unwrap(np.angle(z))
    return float(CubicSpline(x,ph,bc_type=bc)(frd,1))
def main():
    rows=[]
    for c in CASES:
        hp,hc=public_p(F,c); d=public_d(F,c); H=hp+1j*hc; ratio=H/(2*d)
        mf,_,frd_dimless,_=IMRPhenomD.remnant_properties(torch.tensor(c['m1']),torch.tensor(c['m2']),torch.tensor(c['z1']),torch.tensor(c['z2']))
        Msec=(c['m1']+c['m2'])*4.925490947641267e-6; frd=float((frd_dimless/Msec).detach())
        af_pub=float(ls.SimPhenomUtilsPhenomPv2FinalSpin(c['m1'],c['m2'],c['z1'],c['z2'],0.)); wr,_=signed_qnm_220(torch.tensor(af_pub)); frd_pub=float((wr/(2*math.pi)/(mf/(c['m1']+c['m2']))/Msec).detach())
        deriv={str(h):slope(c,frd,h) for h in (1e-2,1e-3,1e-4)}; deriv5={str(h):slope5(c,frd,h) for h in (1e-2,1e-3,1e-4)}; deriv_dimless={str(h):slope(c,frd,h/Msec) for h in (1e-3,1e-4,1e-5)}; deriv_pub={str(h):slope(c,frd_pub,h) for h in (1e-2,1e-3,1e-4)}; deriv5_pub={str(h):slope5(c,frd_pub,h) for h in (1e-2,1e-3,1e-4)}; spline={}
        exact=deriv5['0.0001']
        for lo,hi in ((.5,1.5),(.1,2.)):
            for n in (17,33,65,129,257,513,1025):
                for bc in ('natural','not-a-knot'):
                    spline[f'{lo:g}_{hi:g}_N{n}_{bc}']= {'value':spline_slope(c,frd,lo,hi,n,bc),'absolute_error':abs(spline_slope(c,frd,lo,hi,n,bc)-exact)}
        k=deriv['0.0001']; corr=ratio*np.exp(1j*F*k)
        sparse_f=np.array([20.,100.,300.,400.]); sp_hp,sp_hc=public_p(sparse_f,c); mid_f=np.arange(30.,257.); mid_hp,mid_hc=public_p(mid_f,c); de_hp,de_hc=public_p(F,c,35.); scaled=dict(c,m1=2*c['m1'],m2=2*c['m2']); sf=F/2; sc_hp,sc_hc=public_p(sf,scaled,10.); scf_hp,scf_hc=public_p(sf,scaled,20.)
        rawph=np.unwrap(np.angle(ratio)); rawfit=np.polyfit(F,rawph,1); ph=np.unwrap(np.angle(corr)); fitcoef=np.polyfit(F,ph,1); fit=np.polyval(fitcoef,F)
        def slope_arr(z): return float(np.polyfit(F,np.unwrap(np.angle(z)),1)[0])
        rows.append({'case':c,'f_rd_hz':frd,'public_pv2_final_spin':af_pub,'publicspin_generated_f_rd_hz':frd_pub,'raw_components_vs_D':{'hp_vs_D':rel(hp,d),'hc_vs_minus_iD':rel(hc,-1j*d),'combined_2D':rel(H,2*d),'magnitude_l2':rel(np.abs(H),2*np.abs(d))},'raw_phase_unwrapped':rawph.tolist(),'raw_phase_affine_diagnostic':{'slope':float(rawfit[0]),'intercept':float(rawfit[1])},'derivative_at_clean_frd':deriv,'five_point_derivative_at_clean_frd':deriv5,'dimensionless_grid_derivative':deriv_dimless,'five_point_derivative_at_publicspin_frd':deriv5_pub,'spline_derivatives_at_clean_frd':{'reference':exact,'grids':spline},'derivative_at_publicspin_frd':deriv_pub,'analytic_factor_using_clean_frd':{'factor':'exp(+i*f*k)','slope_used':k,'corrected_phase_slope':float(fitcoef[0]),'corrected_phase_intercept':float(fitcoef[1]),'corrected_complex_l2':rel(corr,np.ones_like(corr)),'max_nonlinear_residual_after_diagnostic_line':float(np.max(np.abs(ph-fit)))},'grid_controls':{'20_512_vs_sparse_hp':rel(hp[[0,80,280,380]],sp_hp),'20_512_vs_sparse_hc':rel(hc[[0,80,280,380]],sp_hc),'20_512_vs_30_256_hp':rel(hp[10:237],mid_hp),'20_512_vs_30_256_hc':rel(hc[10:237],mid_hc),'fref35_hp':rel(de_hp,hp),'fref35_hc':rel(de_hc,hc),'mass2_frequencyhalf_scaled_fref_combined':rel(sc_hp+1j*sc_hc,4*H),'mass2_frequencyhalf_phase_slope_scaled_fref':slope_arr(sc_hp+1j*sc_hc),'mass2_frequencyhalf_phase_slope_fixed_fref':slope_arr(scf_hp+1j*scf_hc),'base_phase_slope':slope_arr(H)}})
    out=pathlib.Path('docs/reviews/2026-09-13-parity-review/pv2-clean'); out.mkdir(parents=True,exist_ok=True)
    report={'provenance':'Public SimIMRPhenomPv2 and SimIMRPhenomDFrequencySequence black boxes; hp/hc copied separately. theta=0 aligned limit expects hp=D and hc=-iD, hence hp+i hc=2D. Final-spin alternate uses public Pv2 final-spin API with clean opaque final mass and generated QNM; diagnostic only.','api_docstrings':{'Pv2':ls.SimIMRPhenomPFrequencySequence.__doc__,'D':ls.SimIMRPhenomDFrequencySequence.__doc__,'Pv2FinalSpin':ls.SimPhenomUtilsPhenomPv2FinalSpin.__doc__},'cases':rows,'versions':{'lal':lal.__version__,'lalsimulation':ls.__version__,'torch':torch.__version__},'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    (out/'time-origin-diagnostic.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
