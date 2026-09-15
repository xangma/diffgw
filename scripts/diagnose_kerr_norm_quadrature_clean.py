#!/usr/bin/env python3
"""Read-only quadrature control for the public Kerr angular norm.

The public callable and the clean truncated-series callable are compared only
as numerical outputs.  No public normalization value is used by runtime code.
"""
import json, math
from pathlib import Path
import numpy as np
from scipy.integrate import quad
import torch
import lal, lalsimulation
from torchwave.approximants.kerr_angular_table_convention import _raw
from torchwave.approximants.kerr_angular import leaver_frobenius_coefficients
from torchwave.approximants.kerr_qnm_generated import signed_qnm_220

def main():
    rows=[]
    for spin in (.1,.3,.7,.9,.95):
        wr, damp=signed_qnm_220(torch.tensor(spin,dtype=torch.float64))
        c=complex(float(spin*wr), float(-spin*damp))
        _, coeff=leaver_frobenius_coefficients(2, torch.tensor(c,dtype=torch.complex128), n_terms=48)
        co=coeff.detach().numpy()
        def own(mu):
            z=(1+mu)/2; total=co[0]
            for n in range(1,len(co)):
                term=co[n]*z**n; total += term
                if abs(term) < .01: break
            return np.exp(c*mu)*(1+mu)**2*total
        A, om=lalsimulation.SimBlackHoleRingdownModeEigenvaluesLeaver(spin/2,2,2,-2)
        def public(mu):
            return complex(lalsimulation.SimBlackHoleRingdownSpheroidalWaveFunctionLeaver(mu,spin/2,2,2,-2,A,om))
        vals={}
        for eps in (1e-6,1e-8,1e-10):
            item={}
            for name, fn in (("own",own),("public",public)):
                val,err=quad(lambda x: abs(fn(x))**2,-1,1,epsabs=eps,epsrel=eps,limit=500)
                item[name]={"integral":val,"error":err}
            vals[str(eps)]=item
        rows.append({"spin":spin,"quadpack":vals})
    out={"spins":[.1,.3,.7,.9,.95],"rows":rows,"note":"quad tolerances are discrete documented controls; public values are never copied or fitted"}
    Path(__file__).with_suffix(".json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))
if __name__ == "__main__": main()
