#!/usr/bin/env python3
"""Clean adaptive embedded 10/21-point Gauss-Kronrod norm diagnostic.

Nodes and weights are generated from orthogonality and moments; no
QUADPACK/LAL integration code or normalization values are imported.
"""
import json, math
from functools import lru_cache
from pathlib import Path
import numpy as np, torch
import sympy as sp
import mpmath as mp
from scipy.integrate import quad
from torchwave.approximants.kerr_angular_table_convention import _raw
from torchwave.approximants.kerr_angular import leaver_frobenius_coefficients
from torchwave.approximants.kerr_qnm_generated import signed_qnm_220
import lalsimulation

@lru_cache(maxsize=1)
def kronrod10_21():
    """Construct K21 from P10*E11 orthogonality and moment weights."""
    mp.mp.dps=80
    x=sp.symbols('x'); p=sp.legendre(10,x)
    es=sp.symbols('e9 e7 e5 e3 e1'); e=x**11+es[0]*x**9+es[1]*x**7+es[2]*x**5+es[3]*x**3+es[4]*x
    eq=[]
    for k in (1,3,5,7,9): eq.append(sp.integrate(sp.expand(p*e*x**k),(x,-1,1)))
    sol=sp.solve(eq,es, rational=True)
    e=sp.Poly(sp.expand(e.subs(sol)),x)
    roots=np.concatenate((np.polynomial.legendre.leggauss(10)[0],np.array([float(sp.re(r)) for r in sp.nroots(e,n=80,maxsteps=200)])))
    roots=np.sort(roots); A=mp.matrix([[mp.mpf(str(r))**k for r in roots] for k in range(21)])
    b=mp.matrix([mp.mpf(2)/(k+1) if k%2==0 else 0 for k in range(21)])
    weights=np.array([float(v) for v in mp.lu_solve(A,b)])
    return np.polynomial.legendre.leggauss(10), (roots,weights)

def adaptive(fn, tol=1e-6):
    (n10,w10),(n21,w21)=kronrod10_21()
    def rule(a,b,n,w):
        x=(b-a)*n/2+(a+b)/2
        return (b-a)/2*np.dot(w,np.asarray([fn(float(q)) for q in x]))
    def leaf(a,b):
        g10=rule(a,b,n10,w10); g21=rule(a,b,n21,w21)
        return [a,b,g21,abs(g21-g10)]
    leaves=[leaf(-1.,1.)]
    budget=lambda total: max(tol, tol*abs(total))
    while sum(q[3] for q in leaves) > budget(sum(q[2] for q in leaves)):
        if len(leaves)>=1000: raise RuntimeError("adaptive K21/G10 exceeded 1000 leaves")
        j=max(range(len(leaves)),key=lambda i: leaves[i][3]); a,b,_,_=leaves.pop(j); mid=(a+b)/2
        leaves.extend((leaf(a,mid),leaf(mid,b)))
    return sum(q[2] for q in leaves),sum(q[3] for q in leaves),len(leaves)

def main():
    (g10n,g10w),(kn,kw)=kronrod10_21()
    assert max(abs(np.dot(kw,kn**k)-(2/(k+1) if k%2==0 else 0)) for k in range(32)) < 2e-12
    assert max(abs(np.dot(g10w,g10n**k)-(2/(k+1) if k%2==0 else 0)) for k in range(20)) < 1e-12
    # Independent quadrature sanity controls (polynomial and smooth function).
    p,pe,_=adaptive(lambda x:x**4,1e-10)
    s,se,_=adaptive(lambda x:math.exp(x),1e-10)
    assert abs(p-.4)<1e-10 and abs(s-(math.e-math.e**-1))<1e-10
    rows=[]
    for spin in (0.,.1,.3,.5,.7,.8,.9,.92,.95,.97):
        wr,damp=signed_qnm_220(torch.tensor(spin,dtype=torch.float64)); c=complex(float(spin*wr),float(-spin*damp))
        _,coeff=leaver_frobenius_coefficients(2,torch.tensor(c,dtype=torch.complex128),n_terms=48); co=coeff.detach().numpy()
        def fn(mu):
            z=(1+mu)/2; total=co[0]
            for n in range(1,len(co)):
                term=co[n]*z**n; total+=term
                if abs(term)<.01: break
            return abs(np.exp(c*mu)*(1+mu)**2*total)**2
        val,err,n=adaptive(fn)
        # Public angular output is normalized; its mu=0 value therefore
        # implies the raw-series norm without importing a public scale.
        A,om=lalsimulation.SimBlackHoleRingdownModeEigenvaluesLeaver(spin/2,2,2,-2)
        pub0=complex(lalsimulation.SimBlackHoleRingdownSpheroidalWaveFunctionLeaver(0.,spin/2,2,2,-2,A,om))
        z0=.5; t0=co[0]
        for nterm in range(1,len(co)):
            term=co[nterm]*z0**nterm; t0+=term
            if abs(term)<.01: break
        own0=abs(t0)
        implied=(own0/abs(pub0))**2
        default,default_err=quad(fn,-1,1,epsabs=1e-6,epsrel=1e-6,limit=500)
        points=[]
        for nterm in range(1,len(co)):
            an=abs(co[nterm])
            if an:
                z=(.01/an)**(1/nterm); cut=2*z-1
                if -1<cut<1: points.append(cut)
        tight,tight_err=quad(fn,-1,1,points=sorted(points),epsabs=1e-10,epsrel=1e-10,limit=500)
        rows.append({"spin":spin,"integral":val,"embedded_error":err,"intervals":n,
                     "public_implied_raw_integral":implied,
                     "relative_integral_error":(val-implied)/implied,
                     "quadpack_default":{"integral":default,"error":default_err,"relative_error":(default-implied)/implied},
                     "quadpack_threshold_split":{"integral":tight,"error":tight_err,"relative_error":(tight-implied)/implied}})
    out={"rows":rows,"rule":"adaptive embedded Gauss-Kronrod K21/G10; error=abs(K21-G10)","tolerance":1e-6,"provenance":"P10/E11 orthogonality and polynomial moments; no public normalization or integration implementation copied"}
    Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
