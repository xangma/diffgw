# XP v223 reference closure

[Pratten et al., Appendix C Eq. (79b)](https://arxiv.org/html/2004.06503v2)
sets the initial Euler opening angle equal to the geometric angle between
\(J\) and \(L_0\). The strict equation-level reference-rotation check in
`tests/test_clean_room_precession_equations.py` retains that identity for the
published 101/102 prescriptions.

The explicit `waveform_v223(..., boundary="v223")` diagnostic follows a
public-LAL compatibility path. Its MSA opening angle uses the nonconserved-spin \(259/72\) coefficient
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
The explicit v223 Torch helper reproduces the public ratio to about \(3\times10^{-15}\).
At the reference frequency, its XAS carrier with the same v223 MSA remnant
override used by the clean XP composition agrees with the ordinary XAS
carrier and public LAL XAS to within \(2\times10^{-12}\); remnant choice does
not explain the ratio.
`tests/test_xp_aligned_boundary_clean.py` compares the two complex ratios
directly, with absolute tolerance \(2\times10^{-12}\), while the independent
equation-level rotation check remains strict. This distinguishes the
published frame identity from the v223 compatibility behavior without
silently broadening the former's tolerance.

## Physical zero-nutation continuation (2026-09-24)

The public Torch XP v223 route now selects a continuous opening angle when at
least one spin has zero transverse components. For
\(\mathbf S=(m_1/M)^2\boldsymbol\chi_1+(m_2/M)^2\boldsymbol\chi_2\),
\(S_\parallel=\mathbf S\cdot\hat{\mathbf L}\), and
\(S_\perp=|\mathbf S-S_\parallel\hat{\mathbf L}|\), it evaluates
\(\cos\beta=(L_\beta+S_\parallel)/
\sqrt{(L_\beta+S_\parallel)^2+S_\perp^2}\).
This is [Pratten et al., Eq. (17)](https://arxiv.org/html/2004.06503v2)
with the native nonconserved-spin 3PN \(L_\beta\); its strict aligned limit
is \(\beta=0\) if \(L_\beta+S_\parallel>0\), or \(\pi\) if the sum is
negative. At vanishing total \(J\), the geometric opening angle is undefined.
The v223 auxiliary-root separation is retained for its compatibility
diagnostic but is not inserted into this physical transverse spin.

The independent [scalar reference](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/xp_msa_regularized_beta.py)
specializes the [Chatziioannou et al. MSA invariants, Eqs. (41), (45),
and (171)](https://arxiv.org/abs/1703.03967v3) to the constant-spin,
one-spin-aligned branch. With \(L_0=\eta/v_\mathrm{ref}\), it obtains
\(S_\mathrm{av}=|\mathbf S|^2\) and \(c_1=\eta S_\parallel\), then evaluates
\(\beta=\operatorname{atan2}(S_\perp,L_\beta+S_\parallel)\) without
small-angle cancellation. The caller supplies \(L_\beta\), so the reference
does not silently equate Newtonian and 3PN orbital angular momentum. The
[nonpublic JAX kernel](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/precession_msa_physical_beta.py)
implements these invariants separately; its frequency-grid convenience
adapter explicitly uses Newtonian \(L=\eta/v\). The
[scalar tests](https://github.com/xangma/diffgw/blob/main/tests/test_xp_msa_regularized_beta_reference.py) and
[JAX tests](https://github.com/xangma/diffgw/blob/main/tests/test_jax_msa_physical_beta.py) check exact and tiny
transverse spins, reversed \(J\), zero-\(J\) handling, JIT, batching, and
selected derivatives. They establish this opening-angle component, not a
whole-waveform JAX implementation or physical validation.

The [Torch XP adapter](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/imrphenom_xp.py)
uses `boundary="physical"` for its supported v223/convention 0, 1, or 6,
`final_spin_mod=3` clean route. It passes the Appendix C observer azimuth as
`phic-phi0`; convention 6 applies its output phase after projection.
Generic two-spin inputs at nonzero `phi0` still use the inherited rotation:
their clean angle transport is not yet validated for that azimuth. The public
JAX XP dispatcher also remains on its inherited route. Opposed-\(J\) rows
retain the existing guard. Thus the physical limit is a scoped Torch XP
behavior, not a claim of complete Torch/JAX or LAL parity.

In the [aligned convention-6 test](https://github.com/xangma/diffgw/blob/main/tests/test_xp_aligned_boundary_clean.py),
the explicit v223 helper retains its strict same-option LAL gate, while the
public physical route has a measured complex-polarization relative L2 between
\(10^{-5}\) and \(5\times10^{-5}\) for that fixture. This residual is reported
rather than fitted away: the two routes encode different opening-angle
contracts. The older reference-frequency XP/XAS ratio above remains evidence
for the explicit binary-compatibility helper, not a paper-derived prediction
for the new physical boundary.
