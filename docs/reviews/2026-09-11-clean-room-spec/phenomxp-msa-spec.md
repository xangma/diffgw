# Clean-Room Mathematical Specification: PhenomXP / XPHM Precession Prescriptions (MSA Variant 223)

## Literature References & Information Boundary
1. **Pratten et al. (2021)**, Phys. Rev. D 103, 104056 ([arXiv:2004.06503v2](https://arxiv.org/html/2004.06503v2)):
   "Let's twist again: computationally efficient models for the gravitational wave signal from precessing black-hole binaries: IMRPhenomXP and IMRPhenomXPHM"
2. **Chatziioannou et al. (2017)**, Phys. Rev. D 95, 104004 ([arXiv:1703.03967](https://arxiv.org/abs/1703.03967)):
   "Constructing Gravitational Waves from Generic Spin-Precessing Compact Binary Inspirals"
3. **Chatziioannou et al. (2013)**, Phys. Rev. D 88, 063011 ([arXiv:1307.4418](https://arxiv.org/abs/1307.4418)):
   "Analytical Gravitational Waveforms for Generic Precessing Compact Binaries"

**STRICT CLEAN-ROOM RULES**:
- Implement ONLY from the mathematics described herein and standard post-Newtonian theory.
- DO NOT inspect, read, copy, or port any LAL, LALSimulation, or PyCBC source code or internal routines.
- DO NOT fit coefficients or offsets to empirical oracle fixtures.
- Pure PyTorch native implementation with automatic fallback and full autograd support.

---

## 1. System Coordinates & Conserved Quantities
Units: Geometric units where G = c = M = 1, with component masses m1 >= m2, mass ratio q = m1/m2 >= 1, delta = (m1 - m2)/M, eta = m1 * m2 / M^2.
Component spins: S1 = m1^2 * chi1, S2 = m2^2 * chi2.

At reference frequency f_ref, the orbital velocity is v0 = (pi * M * f_ref)^(1/3), and the orbital angular momentum is L0 = L(v0) * z_L0.
The total angular momentum vector is:
J0 = L0 + S1 + S2
The conserved projected effective spin is:
xi = (1 + q) * S1 . L_hat + (1 + 1/q) * S2 . L_hat = (1 + q) * S1z + (1 + 1/q) * S2z

---

## 2. 3PN Spinning Orbital Angular Momentum L(v)
From Pratten et al. (2020) Eq. 18 and Appendix C:
For orbital velocity v = (pi * M * f)^(1/3) and x = v^2:
L(v) = (eta / v) * [ 1 + L1 * v^2 + L_15_SO * v^3 + L2 * v^4 + L_25_SO * v^5 + L3 * v^6 + L_35_SO * v^7 ]
where:
L1 = 3/2 + eta / 6
L2 = 27/8 - 19/8 * eta + eta^2 / 24
L3 = 135/16 + (41 * pi^2 / 24 - 6889/144) * eta + 31/24 * eta^2 + 7/1296 * eta^3
Spin-orbit contributions:
S_ell = S1z + S2z,  Sigma_ell = m2 * chi1z - m1 * chi2z
L_15_SO = -35/6 * S_ell - 5/2 * delta * Sigma_ell
L_25_SO = (-77/8 + 427/72 * eta) * S_ell + delta * (-21/8 + 35/12 * eta) * Sigma_ell
L_35_SO = (-405/16 + 1101/16 * eta - 29/16 * eta^2) * S_ell + delta * (-81/6 + 117/4 * eta - 15/16 * eta^2) * Sigma_ell

---

## 3. Spin Magnitude Roots (S_-^2, S_+^2, S_3^2)
The magnitude S^2 = |S1 + S2|^2 obeys the cubic:
(dS^2 / dt)^2 = -A^2 * [ (S^2)^3 + B * (S^2)^2 + C * (S^2) + D ]
where (Chatziioannou et al. 2017 Appendix A):
A = -3 / (2 * sqrt(eta)) * v^6 * (1 - xi * v)
B = (L^2 + S1^2)*q + 2*L*xi - 2*J^2 - S1^2 - S2^2 + (L^2 + S2^2)/q
C = (J^2 - L^2)^2 - 2*L*xi*(J^2 - L^2) - 2*((1-q)/q)*(S1^2 - q*S2^2)*L^2 + 4*eta*L^2*xi^2 - 2*delta*(S1^2 - S2^2)*xi*L + 2*((1-q)/q)*(q*S1^2 - S2^2)*J^2
D = ((1-q)/q)*(S2^2 - q*S1^2)*(J^2 - L^2)^2 + (delta^2 / eta)*(S1^2 - S2^2)^2 * L^2 + 2*delta*L*xi*(S1^2 - S2^2)*(J^2 - L^2)

Solve the depressed cubic for the 3 real roots sorted in ascending order:
S_3^2 <= S_-^2 <= S_+^2

Physical Validity Gate:
- Roots must be real and ordered: S_3^2 <= S_-^2 <= S_+^2.
- S_+^2 >= S_-^2 >= 0.
- If S_+^2 - S_-^2 < 1e-9 (collinear / non-precessing) or if q - 1 < 1e-4 (equal mass singular limit), trigger NNLO fallback per Pratten et al. Table 3.

---

## 4. Precession Constants & Evolution
Initial roots S_{+, 0}^2, S_{-, 0}^2 evaluated at v0:
S_av^2 = 0.5 * (S_{+, 0}^2 + S_{-, 0}^2)
c1 = 0.5 * v0 * (J0^2 - L(v0)^2 - S_av^2)
Total angular momentum evolution along the inspiral:
J(v) = sqrt(L(v)^2 + 2*c1/v + S_av^2)
Opening angle of the precession cone:
cos_beta(f) = (J(v)^2 + L(v)^2 - S_av^2) / (2 * J(v) * L(v))
beta(f) = arccos(clamp(cos_beta, -1, 1))

---

## 5. Multiple Scale Analysis (MSA) Euler Angles

### A. Precession Phase psi(v)
From Chatziioannou et al. (2017) Eq. 65:
psi(v) = psi0 - (3 * g0 / 4) * delta * v^(-3) * [ 1 + psi1 * v + psi2 * v^2 ]
where psi0 is set such that psi(v0) = 0:
psi1 = 3 * (2 * xi * eta^2 - c1) / (eta * delta^2)
psi2 = 3 * g2 / g0 + 3 / (2 * eta^3) * [ 2*Delta - 2*(eta^2 / delta^2)*S_av^2 - 10*(eta / delta^4)*c1^2 + 2*(eta^2 / delta^2)*((7 + 6*q + 7*q^2)/(1-q)^2)*c1*xi - (eta^3 / delta^2)*((3 + 4*q + 3*q^2)/(1-q)^2)*xi^2 + (eta / (1-q)^2)*(q*(2+q)*S1^2 + (1+2*q)*S2^2) ]
Delta = sqrt( [ c1^2*eta/(q*delta^4) - 2*c1*eta^3*(1+q)/(q*delta^4)*xi - eta^2/delta^4*(delta^2*S1^2 - eta^2*xi^2) ] * [ c1^2*eta^2/delta^4 - 2*c1*eta^3*(1+q)/delta^4*xi - eta^2/delta^4*(delta^2*S2^2 - eta^2*xi^2) ] )

### B. Dissipative Coefficients g_n
g0 = 5 / (96 * eta)
a0 = 96 * eta / 5
a2 = -743/336 - 11/4 * eta
<S1_ell> = (c1 * (1 + q) - q * eta * xi) / (eta * (1 - q^2))
<S2_ell> = -q * (c1 * (1 + q) - eta * xi) / (eta * (1 - q^2))
beta3 = (113/12 + 25/(4*q)) * <S1_ell> + (113/12 + 25*q/4) * <S2_ell>
a3 = 4 * pi - beta3
beta5 = ( (31319/1008 - 1159/24*eta) + (1/q)*(809/84 - 281/8*eta) ) * <S1_ell> + ( (31319/1008 - 1159/24*eta) + q*(809/84 - 281/8*eta) ) * <S2_ell>
sigma4 = (247/48 * (S_av^2/2 - (S1^2+S2^2)/2) - 721/48 * <S1_ell> * <S2_ell>) / eta + (233/96 * S1^2 - 719/96 * <S1_ell>^2) / m1^2 + (233/96 * S2^2 - 719/96 * <S2_ell>^2) / m2^2
a4 = 34103/18144 + 13661/2016 * eta + 59/18 * eta^2 - sigma4
a5 = -4159/672 * pi - 189/8 * pi * eta - beta5

g2 = -a2 / a0
g3 = -a3 / a0
g4 = -(a4 - a2^2) / a0
g5 = -(a5 - 2 * a3 * a2) / a0

### C. Azimuthal Angle alpha(v)
alpha(v) = alpha_{-1}(v) + alpha_0(v)
alpha_{-1}(v) = sum_{n=0}^5 <Omega_z>^{(n)} * phi_z^{(n)}(v)
Coefficients Omega_{z, n}:
R_m = S_+^2 - S_-^2
c_p = S_+^2 * eta^2 - c1^2
c_m = S_-^2 * eta^2 - c1^2
a1 = 1/2 + 3/4 * eta
a2 = -3 / (4 * eta) * xi
a_d = (-3 * (S1^2 - S2^2) * eta * delta + 3 * (c1 / eta) * (c1 - 2 * xi * eta^2)) / (4 * sqrt(c_p * c_m))
c_d = 3/128 * R_m^2 / (eta * sqrt(c_p * c_m))
h_d = (c1 / eta^2) * (1 - (c_p + c_m) / (2 * sqrt(c_p * c_m)))
f_d = (c_p + c_m) / (8 * eta^4) * (1 - (c_p + c_m) / (2 * sqrt(c_p * c_m)))

Omega_{z, 0} = a1 + a_d
Omega_{z, 1} = a2 - a_d * xi - a_d * h_d
Omega_{z, 2} = a_d * h_d * xi + c_d - a_d * f_d + a_d * h_d^2
Omega_{z, 3} = (a_d * f_d - c_d - a_d * h_d^2) * (xi + h_d) + a_d * f_d * h_d
Omega_{z, 4} = (c_d + a_d * h_d^2 - 2 * a_d * f_d) * (h_d * xi + h_d^2 - f_d) - a_d * f_d^2
Omega_{z, 5} = (c_d - a_d * f_d + a_d * h_d^2) * f_d * (xi + 2 * h_d) - (c_d + a_d * h_d^2 - 2 * a_d * f_d) * h_d^2 * (xi + h_d) - a_d * f_d^2 * h_d

<Omega_z>^{(0)} = 3 * g0 * Omega_{z, 0}
<Omega_z>^{(1)} = 3 * g0 * Omega_{z, 1}
<Omega_z>^{(2)} = 3 * (g0 * Omega_{z, 2} + g2 * Omega_{z, 0})
<Omega_z>^{(3)} = 3 * (g0 * Omega_{z, 3} + g2 * Omega_{z, 1} + g3 * Omega_{z, 0})
<Omega_z>^{(4)} = 3 * (g0 * Omega_{z, 4} + g2 * Omega_{z, 2} + g3 * Omega_{z, 1} + g4 * Omega_{z, 0})
<Omega_z>^{(5)} = 3 * (g0 * Omega_{z, 5} + g2 * Omega_{z, 3} + g3 * Omega_{z, 2} + g4 * Omega_{z, 1} + g5 * Omega_{z, 0})

Functions phi_z^{(n)}(v):
l1 = ln(c1 + J * eta + L * eta)
l2 = ln(c1 + J * sqrt(S_av^2) * v + S_av^2 * v)
phi_z^{(0)} = (J / eta^4) * (c1^2 / 2 - c1 * eta^2 / (6 * v) - S_av^2 * eta^2 / 3 - eta^4 / (3 * v^2)) - (c1 / (2 * eta)) * (c1^2 / eta^4 - S_av^2 / eta^2) * l1
phi_z^{(1)} = -J / (2 * eta^2) * (c1 + eta * L) + 1 / (2 * eta^3) * (c1 - eta^2 * S_av^2) * l1
phi_z^{(2)} = -J + sqrt(S_av^2) * l2 - (c1 / eta) * l1
phi_z^{(3)} = J * v - eta * l1 + (c1 / sqrt(S_av^2)) * l2
phi_z^{(4)} = J / (2 * S_av^2) * v * (c1 + v * S_av^2) - 1 / (2 * (S_av^2)^(3/2)) * (c1^2 - eta^2 * S_av^2) * l2
phi_z^{(5)} = -J * v * (c1^2 / (2 * (S_av^2)^2) - c1 * v / (6 * S_av^2) - v^2 / 3 - eta^2 / (3 * S_av^2)) + c1 / (2 * (S_av^2)^(5/2)) * (c1^2 - eta^2 * S_av^2) * l2

### D. Coprecessing Angle gamma(v) = -epsilon(v)
gamma_{-1}(v) = eta * v^(-3) * sum_{n=0}^5 <Omega_zeta>^{(n)} * v^n
where:
g_d = 3/64 * R_m^2 / eta^3 * (c1 - eta^2 * xi) / sqrt(c_p * c_m)
Omega_{zeta, 0} = Omega_{z, 0}
Omega_{zeta, 1} = Omega_{z, 1} + (c1 / eta^2) * Omega_{z, 0}
Omega_{zeta, 2} = Omega_{z, 2} + (c1 / eta^2) * Omega_{z, 1}
Omega_{zeta, 3} = Omega_{z, 3} + (c1 / eta^2) * Omega_{z, 2} + g_d
Omega_{zeta, 4} = Omega_{z, 4} + (c1 / eta^2) * Omega_{z, 3} - g_d * xi - g_d * h_d
Omega_{zeta, 5} = Omega_{z, 5} + (c1 / eta^2) * Omega_{z, 4} + g_d * h_d * xi + g_d * (h_d^2 - f_d)

<Omega_zeta>^{(0)} = -g0 * Omega_{zeta, 0}
<Omega_zeta>^{(1)} = -1.5 * g0 * Omega_{zeta, 1}
<Omega_zeta>^{(2)} = -3 * (g0 * Omega_{zeta, 2} + g2 * Omega_{zeta, 0})
<Omega_zeta>^{(3)} = 3 * (g0 * Omega_{zeta, 3} + g2 * Omega_{zeta, 1} + g3 * Omega_{zeta, 0})
<Omega_zeta>^{(4)} = 3 * (g0 * Omega_{zeta, 4} + g2 * Omega_{zeta, 2} + g3 * Omega_{zeta, 1} + g4 * Omega_{zeta, 0})
<Omega_zeta>^{(5)} = 1.5 * (g0 * Omega_{zeta, 5} + g2 * Omega_{zeta, 3} + g3 * Omega_{zeta, 2} + g4 * Omega_{zeta, 1} + g5 * Omega_{zeta, 0})

---

## 6. Frame Alignment & Offsets (Convention 1 Default)
At reference frequency f_ref:
J0 = (S1x + S2x, S1y + S2y, L0 + S1z + S2z)
theta_JL0 = arccos(J0z / |J0|),  phi_JL0 = atan2(J0y, J0x)
Line of sight in L0:
N_hat_L0 = (sin(iota) * sin(phi_ref), sin(iota) * cos(phi_ref), cos(iota))
Rotate to J' frame:
N_hat_Jprime = R_y(-theta_JL0) * R_z(-phi_JL0) * N_hat_L0
kappa = atan2(N_hat_Jprime_y, N_hat_Jprime_x)

Offsets (Table 4 Convention 1):
alpha_0_XP = pi - kappa
beta_0 = theta_JL0
epsilon_0_XP = phi_JL0 - pi
alpha_0_Pv2 = pi - phi_JL0

alpha_offset = alpha_MSA(f_ref) - alpha_0_Pv2
epsilon_offset = epsilon_MSA(f_ref) - epsilon_0_XP

alpha(f) = alpha_MSA(f) - alpha_offset
epsilon(f) = epsilon_MSA(f) - epsilon_offset
gamma(f) = -epsilon(f)

Polarization basis rotation zeta:
x_N = R_z(-kappa) * R_y(-theta_JL0) * R_z(-phi_JL0) * (-cos(iota)*sin(phi_ref), -cos(iota)*cos(phi_ref), sin(iota))^T
theta_JN = arccos(clamp(J0 . N_hat / |J0|, -1, 1))
x_N_prime = (cos(theta_JN), 0, -sin(theta_JN))
y_N_prime = (0, 0, 1)
zeta = atan2(x_N . y_N_prime, x_N . x_N_prime)

---

## 7. Multipolar Wigner Twist-Up
For all active modes (l, m) and m' in [-l, l]:
h_{lm}^J(f) = sum_{m'=-l}^l d^l_{m, m'}(beta(f)) * exp(-i * m' * gamma(f)) * h_{l, m'}^{coprec}(f) * exp(-i * m * alpha(f))

where h_{l, m'}^{coprec}(f) is from clean-room IMRPhenomXHM.
Project with -2Y_{lm}(theta_JN, 0) and rotate polarizations:
h_+ - i h_x = exp(2i * zeta) * sum_{l, m} -2Y_{lm}(theta_JN, 0) * h_{lm}^J(f)
h_+(f) = Re(h_+ - i h_x)
h_x(f) = -Im(h_+ - i h_x)
