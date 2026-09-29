"""Independent high-precision reference-frame calculation for the 37.3 Hz case.

Pratten et al. (2021), Appendix C, Eqs. (C4)-(C16), defines the J frame
and its reference Euler angles. This uses the model's declared 3PN L_ref
prescription and a vector projection, rather than its rotation code.
"""

import mpmath as mp

mp.mp.dps = 60
D = mp.mpf
m1, m2, f_ref = D(36), D(29), D("37.3")
s1 = [D("0.3"), D("0.2"), D("-0.1")]
s2 = [D("0.1"), D("-0.2"), D("0.2")]
iota, phi_ref = D("0.6"), D(0)
M = m1 + m2
eta = m1 * m2 / M**2
delta = (m1 - m2) / M
msun_sec = D("6.67430e-11") * D("1.9884098706980507e30") / D(299792458)**3
v = (mp.pi * M * msun_sec * f_ref)**(D(1)/3)
L2 = D(3)/2 + eta/6
L3 = 5 * (s1[2]*(-2-2*delta+eta) + s2[2]*(-2+2*delta+eta))/6
L4 = (81 + (-57+eta)*eta)/24
L5 = -7 * (s1[2]*(72+delta*(72-31*eta)+eta*(-121+2*eta))
           + s2[2]*(72+eta*(-121+2*eta)+delta*(-72+31*eta)))/144
L6 = (10935 + eta*(-62001 + eta*(1674+7*eta) + 2214*mp.pi**2))/1296
L_ref = eta/v * (1 + L2*v**2 + L3*v**3 + L4*v**4 + L5*v**5 + L6*v**6)
J = [((m1/M)**2)*s1[k] + ((m2/M)**2)*s2[k] for k in range(3)]
J[2] += L_ref
N = [mp.sin(iota)*mp.sin(phi_ref), mp.sin(iota)*mp.cos(phi_ref), mp.cos(iota)]
dot = lambda a, b: sum(x*y for x, y in zip(a, b))
norm = lambda a: mp.sqrt(dot(a, a))
cross = lambda a, b: [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
zj = [x/norm(J) for x in J]
xj_unnorm = [N[k]-dot(N,zj)*zj[k] for k in range(3)]
xj = [x/norm(xj_unnorm) for x in xj_unnorm]
yj = cross(zj,xj)
L0 = [D(0),D(0),D(1)]
alpha = mp.atan2(dot(L0,yj),dot(L0,xj))
beta = mp.acos(dot(L0,zj))
gamma = mp.pi - mp.atan2(J[1],J[0])
for label, value in [('L_ref',L_ref),('alpha_ref',alpha),('beta_ref',beta),('gamma_ref',gamma)]:
    print(label, mp.nstr(value, 35))
