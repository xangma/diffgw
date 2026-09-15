"""Compare analytic and stabilized quadratic PA initial values on cached grids."""
import hashlib, json, shlex, sys
from pathlib import Path
import numpy as np
import torch
from torchwave.approximants.eob.teob_conditioned_waveform_clean import _make_hamiltonian, _make_reaction
from torchwave.approximants.eob.teob_postadiabatic_clean import alternating8pa

def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(path, m1, m2):
    z=np.load(path); d=torch.float64
    h=_make_hamiltonian(torch.tensor(m1,dtype=d),torch.tensor(m2,dtype=d))
    r=torch.as_tensor(z['pa_radius'],dtype=d); n_cached=r.numel()
    kwargs=dict(omega_start=torch.as_tensor(z['omega_start'],dtype=d),r_min=14.0,dr=0.1,pa_order=8)
    reaction=_make_reaction('leading-order',h)
    out={"cache":str(path),"sha256":sha256(path),"physical_masses_msun":[m1,m2],"pa_knots":len(r),
         "pa_kwargs":{"omega_start":float(kwargs['omega_start']),"r_min":14.0,"dr":0.1,"pa_order":8},
         "reaction":"_make_reaction('leading-order', hamiltonian), include_horizon_absorption=True",
         "initializer_policies":["analytic","quadratic-value"]}
    for policy in ('analytic','quadratic-value'):
        q=alternating8pa(h,reaction,include_horizon_absorption=True,initializer_policy=policy,**kwargs)
        s=q.state[0]
        cached=torch.stack((torch.as_tensor(z['pa_radius'][0],dtype=d),torch.zeros((),dtype=d),
                            torch.as_tensor(z['pa_pr_star'][0],dtype=d),torch.as_tensor(z['pa_p_phi'][0],dtype=d)))
        cached_all=torch.stack((torch.as_tensor(z['pa_radius'],dtype=d),torch.as_tensor(z['pa_phase'],dtype=d),
                                torch.as_tensor(z['pa_pr_star'],dtype=d),torch.as_tensor(z['pa_p_phi'],dtype=d)),dim=-1)
        if q.state.shape != cached_all.shape or q.time.shape != torch.as_tensor(z['pa_time']).shape:
            raise RuntimeError(f"replayed PA grid shape mismatch: {q.state.shape}/{q.time.shape} vs {cached_all.shape}/{z['pa_time'].shape}")
        if not torch.equal(q.state[:, 0], r) :
            raise RuntimeError("replayed radial grid differs from cached PA grid")
        n=n_cached; state_delta=q.state-cached_all
        cached_time=torch.as_tensor(z['pa_time'],dtype=d); time_delta=q.time-cached_time
        out[policy]={"status":q.status,"initial_state":s.detach().numpy().tolist(),
                     "max_state_abs_from_cached":state_delta.abs().max().item(),
                     "state_relative_l2_from_cached":(torch.linalg.vector_norm(state_delta)/torch.linalg.vector_norm(cached_all)).item(),
                     "max_time_abs_from_cached":time_delta.abs().max().item(),
                     "time_relative_l2_from_cached":(torch.linalg.vector_norm(time_delta)/torch.linalg.vector_norm(cached_time)).item(),"replayed_knots":n,"cached_knots":int(cached_all.shape[0]),"max_state_abs_by_component":state_delta.abs().max(dim=0).values.detach().numpy().tolist()}
        if policy=='analytic': analytic_state,analytic_time=q.state,q.time
        else:
            ds=q.state-analytic_state; dt=q.time-analytic_time
            out['quadratic_vs_analytic']={"max_state_abs":ds.abs().max().item(),"state_relative_l2":(torch.linalg.vector_norm(ds)/torch.linalg.vector_norm(analytic_state)).item(),"max_time_abs":dt.abs().max().item(),"time_relative_l2":(torch.linalg.vector_norm(dt)/torch.linalg.vector_norm(analytic_time)).item()}
    out['quadratic_minus_analytic'] = max(abs(a-b) for a,b in zip(out['quadratic-value']['initial_state'],out['analytic']['initial_state']))
    return out

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--native',type=Path,required=True); p.add_argument('--catalog',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    p0={str(a.native):sha256(a.native),str(a.catalog):sha256(a.catalog)}
    deps=[Path(__file__),Path('src/torchwave/approximants/eob/teob_postadiabatic_clean.py'),Path('src/torchwave/approximants/eob/teob_conditioned_waveform_clean.py'),Path('src/torchwave/approximants/eob/hamiltonian.py'),Path('src/torchwave/approximants/eob/radiation_reaction.py'),Path('src/torchwave/approximants/eob/teob_radiation_clean.py'),Path('src/torchwave/approximants/eob/teob_vacuum_driver_clean.py')]
    report={'command':shlex.join([sys.executable,*sys.argv]),
            'metric_definition_note':'Stage-aware FD report uses concatenated plus/cross polarization norms; complex-combined band diagnostics are not numerically interchangeable.',
            'dependency_sha256':{str(p):sha256(p) for p in deps},'cases':[run(a.native,30.,20.),run(a.catalog,1.4,1.4)],'input_sha256_before':p0}
    report['input_sha256_after']={str(a.native):sha256(a.native),str(a.catalog):sha256(a.catalog)}
    report['inputs_unchanged']=report['input_sha256_before']==report['input_sha256_after']
    a.output.write_text(json.dumps(report,indent=2)+'\n')
