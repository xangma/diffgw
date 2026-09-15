# Clean-room MSA equation audit

This audit compares the fresh TorchWave modules with the cached primary source
`/tmp/torchwave-clean-room-precession/1703.03967v3-paper.tex`; no legacy MSA or
LAL implementation was inspected.

The invariant construction in `precession_msa_initial_clean.py` sorts the
masses, converts dimensionless spins to (S_i=(m_i/M)^2\chi_i), computes

\[
 \mathbf J=\mathbf L+\mathbf S_1+\mathbf S_2,\qquad
 \xi=(1+q)\mathbf S_1\cdot\hat L+(1+q^{-1})\mathbf S_2\cdot\hat L,
\]

matching the paper's conserved-quantity definition (lines 373--382). Its
cubic coefficients are the displayed (B,C,D) expressions (paper lines
1403--1413), and roots are ordered (S_3^2\le S_-^2\le S_+^2).

The default assembly uses the paper's leading prescription

\[
 S_{\rm av}^2=(S_{+,0}^2+S_{-,0}^2)/2
\]

(paper lines 698--706), then sets
`c1 = v0*(J0**2-L0**2-Sav)/2`, exactly Eq. (Jsol). The optional exact
elliptic selector retains the full Eq. (26) average; it is explicitly an
optional extension, while the PN truncation remains approximate.

The initial Jacobi phase uses the inverse-amplitude branch and the sign of the
triple product. This agrees with the paper's sign relation
`sign(sin(phi')) = sign((Lhat x S1).S2) = sign(-dS^2/dt)` (line 535).

The clean PN phase coefficients use the specified limiting-root construction:
`psi1=3/2*(r1-xi)` and `psi2=3*(r2-xi*r1+g2/g0)`. The first is algebraically
identical to the paper's printed `3*(2*xi*eta**2-c1)/(eta*delta_m**2)`
(line 1477). The second is the root form required by the clean-room
specification. It is a derived implementation of the published expansion,
not an assertion of term-by-term identity with the displayed expression
involving `Delta` and spin terms (lines 1479--1492), which also contains a
transcription issue addressed by the clean derivation. Anchoring is consistent with Eq. (psi_rr)
(lines 804--808): only `raw(v)-raw(v0)` is added to the reference phase.

The secular coefficients are evaluated with the fixed reference roots
`sp0,sm0`, while the evolving roots are used only in the instantaneous spin
oscillation and rational correction. This matches the MSA organization: the
paper promotes roots to radiation-reaction functions in Eq. (S-sol), but its
PN-reexpanded averaged coefficients are formed from the reference expansion
(lines 1591--1626).

The MSA paper explicitly defines evolution with (L=\eta/v) (lines 682--683).
The separate Sec. PN_L of Pratten et al., arXiv:2004.06503v2, introduces
PN aligned-spin estimates for the opening angle beta. These are different
primary sources. The clean `beta_pn=True` output is
compared against the same public target using its PN-L path; any resulting
first-frequency difference remains a measured discrepancy and is not resolved
by describing the MSA roots as Newtonian.

## Appendix-F zeta factor

The paper's exact no-radiation expression has `phi_z/J` and coefficients
divided by `dot(psi)` (lines 571--577). Its displayed first MSA correction for
zeta is Eq. (zeta0), lines 1702--1710:

\[
 \zeta_0={A_\theta\over\dot\psi}(C_\phi+D_\phi)+
 {2d_0B_\theta\over\dot\psi}\left({C_\phi\over s_d-d_2}-{D_\phi\over s_d+d_2}\right).
\]

The clean rational integral additionally contains the mean-zero contribution
`(a+c4/d4)*B_theta*(-sin(2 psi)/4)*J/dot(psi)`, from multiplying the constant
part of `dot(phi_z)/J` by `B_theta sin^2(psi)`. Appendix-B defines the partial
fraction coefficients from the ratios (c_i/d_i) and does not display an
absorption of this term into (C_\phi,D_\phi). Its scaling is (O(v)) in the
large-(L) limit, so it is within the first oscillatory correction rather than
an obviously higher-order remainder. The separate
`diagnose_msa_zeta_constant_term_clean.py` report evaluates the literal
Eq. (zeta0) omission as a diagnostic only; runtime remains unchanged.

The public version-isolation experiment is recorded in
`msa-public-version-isolation-reference.json`. It uses the table statement
that version 224 is version 220 with the Eq. (phiz0)/(zeta0) prescription from
223 (Pratten et al., arXiv:2004.06503v2, table lines 1949--1953). For 35/25,
the anchored public 220-minus-224 alpha difference is approximately zero
(6.6e-14); it is also approximately zero for 45/15 (8.9e-16). The diagnostic
compares that difference with a zero clean difference. This is a null control,
not evidence that the clean oscillatory correction reproduces either public
version. The corresponding gamma residuals are 1.71 and 0.143 rad.
The no-J and omitted-zeta-Q variants do not improve this jointly. The
223-minus-224 arrays are retained in the report with first- and second-
finite-difference maxima, so smoothness is measured rather than assumed. The
paper's convention is ε=-ζ (the MSA paper's ζ is the waveform third Euler
angle), whereas the public callable labels its second returned series
`gamma`; the report therefore gives both signs when comparing a clean zeta
candidate.

Mathematical checks run with the configured Python environment: the focused
clean MSA suite reports 10 passed. The independent elliptic quadrature checks
at (m=0.2,0.5,0.8) agree with the helper to at most (4.5\times10^{-16}).
