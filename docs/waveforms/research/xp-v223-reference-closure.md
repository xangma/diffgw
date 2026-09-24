# XP v223 reference closure

[Pratten et al., Appendix C Eq. (79b)](https://arxiv.org/html/2004.06503v2)
sets the initial Euler opening angle equal to the geometric angle between
\(J\) and \(L_0\). The strict equation-level reference-rotation check in
`tests/test_clean_room_precession_equations.py` retains that identity for the
published 101/102 prescriptions.

The native XP v223 composition follows a separate public-LAL compatibility
path. Its MSA opening angle uses the nonconserved-spin \(259/72\) coefficient
printed in [Blanchet, Buonanno and Faye, Eq. (6.9)](https://arxiv.org/pdf/gr-qc/0605140v4).
Pratten's [Appendix G Eq. (123g)](https://arxiv.org/html/2004.06503v2)
instead prints \(427/72\) in a different conserved-spin orbital-momentum
series. The \(259/75\) coefficient used for v223 source-frame geometry is a
compatibility value inferred from the public implementation; it is not stated
by either paper. These expressions must not be conflated as one published
equation.

For the test binary \((m_1,m_2)=(40,20)\,M_\odot\),
\(\boldsymbol\chi_1=(0.3,0.2,0.4)\),
\(\boldsymbol\chi_2=(0,0,-0.2)\), \(f_\mathrm{ref}=40\) Hz,
\(\iota=0.7\), and \(\phi_\mathrm{ref}=0.3\), public LAL XP/XAS has a
reference-frequency complex strain ratio differing from unity by
\(1.4757043\times10^{-6}\) in plus and \(1.9991717\times10^{-6}\) in cross.
The clean Torch path reproduces the public ratio to about \(3\times10^{-15}\).
At the reference frequency, its XAS carrier with the same v223 MSA remnant
override used by the clean XP composition agrees with the ordinary XAS
carrier and public LAL XAS to within \(2\times10^{-12}\); remnant choice does
not explain the ratio.
`tests/test_xp_aligned_boundary_clean.py` compares the two complex ratios
directly, with absolute tolerance \(2\times10^{-12}\), while the independent
equation-level rotation check remains strict. This distinguishes the
published frame identity from the v223 compatibility behavior without
silently broadening the former's tolerance.
