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
selected derivatives. They establish this opening-angle component independently
of either waveform frontend; they do not establish physical validation.

The [Torch XP adapter](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/imrphenom_xp.py)
uses `boundary="physical"` for its supported v223/convention 0, 1, or 6,
`final_spin_mod=3` clean route. It passes the Appendix C observer azimuth as
`phic-phi0`; convention 6 applies its output phase after projection.
Valid unequal-mass generic two-spin inputs at nonzero `phi0` now use that
same clean chart. Opposed-\(J\) rows retain the existing guard. The physical
limit is a scoped XP behavior,
not a claim of complete precession-domain or LAL parity.

The [public JAX XP adapter](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/imrphenom_xp.py)
selects the constant-spin MSA limit when either spin has exactly zero
transverse components, for v223, conventions 0/1/6, and
`final_spin_mod=3`. Its
[intrinsic kernel](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/precession_msa_v223_zero_nutation.py)
uses the constant-spin reduction of [Chatziioannou et al., Eqs. (41), (45),
(65) and Appendices D/F](https://arxiv.org/abs/1703.03967v3) for secular
angles and the MSA remnant prescription. The opening uses the unsplit
Pratten Eq. (17) spin vector above. The
[projection](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/precession_msa_v223_one_spin_projection.py)
combines the source geometry from Pratten Appendix C, the quadrupole rotation
from Eqs. (8)–(13), and the declared v223/Table-IV chart choices. Public
`phic-phi0` supplies the source azimuth, and convention 6 rotates the
completed strain by \(e^{2i\,\mathrm{phic}}\). Valid unequal-mass generic
two-spin rows use the distinct-root MSA route at either observer azimuth;
opposed-\(J\) rows retain inherited JAX behavior. Zero-frequency bins are
evaluated at a safe positive frequency and then masked to zero.

The [JAX public-route tests](https://github.com/xangma/diffgw/blob/main/tests/test_jax_public_xp_zero_nutation.py)
compare selected complete complex polarizations with the Torch physical
route, including exact and tiny transverse spin, mass exchange, equal mass,
nonzero `phi0`, conventions 0/1/6, zero bins, batching, JIT, and a selected
gradient. The [intrinsic tests](https://github.com/xangma/diffgw/blob/main/tests/test_jax_clean_v223_zero_nutation.py)
separately compare secular angles and remnant with Torch and the physical
opening with the scalar reference. These checks verify the scoped JAX
implementation against its declared equations and Torch composition; they do
not qualify generic two-spin dispatch or establish physical accuracy.

The same endpoint must also satisfy the minimal-rotation relation
\(d\gamma=-\cos\beta\,d\alpha\) in
[Pratten et al., Eq. (7)](https://arxiv.org/html/2004.06503v2). At
\(\beta=0\), \(\alpha+\gamma\) is constant. The auxiliary-root v223 angles
instead accumulate a nonzero aligned \(\alpha+\gamma\) drift. The public
physical route removes the aligned same-series drift, anchored at the angle
reference frequency, and makes its final spin tend to the XAS aligned fit.
The [scalar reference](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/xp_msa_regularized_beta.py)
specifies the exact Eq. (7) endpoint, while the
[Torch test](https://github.com/xangma/diffgw/blob/main/tests/test_xp_physical_beta_torch.py)
checks the production angles against it. The XAS carrier preserves an omitted
`f_ref` as `None`; the angle kernel still uses its 20 Hz default.

To continue this correction into small one-spin transverse values, its weight
is \(\Delta/(\Delta+T)\), where \(\Delta=10^{-9}\) is the existing v223
auxiliary \(S^2\) root gap and
\(T=\sum_i(m_i/M)^4(\chi_{ix}^2+\chi_{iy}^2)\). This is a numerical
regularization of the auxiliary-root artifact, not an independent published
waveform equation. It approaches one at alignment and zero as the physical
transverse spin resolves the root gap. In the \((36,22)M_\odot\),
\((\chi_{1z},\chi_{2z})=(0.35,-0.25)\) test, exact XP/XAS complex-strain
maximum relative differences are below \(5\times10^{-15}\) for omitted,
20 Hz, and 40 Hz `f_ref`. The one-spin \(\chi_{1x}=10^{-8}\) and
\(10^{-10}\) relative L2 differences from exact alignment are about
\(2.1\times10^{-9}\) and \(2.7\times10^{-11}\), respectively. A finite
one-spin \(\chi_{1x}=0.3\) retains a \(1.13\times10^{-8}\) L2 difference
from the pinned LAL diagnostic.

In the [aligned convention-6 test](https://github.com/xangma/diffgw/blob/main/tests/test_xp_aligned_boundary_clean.py),
the explicit v223 helper retains its strict same-option LAL gate, while the
public physical route differs from the pinned v223 LAL output by about
\(0.117826\) relative complex-polarization L2 for that fixture. This is the
intentional consequence of enforcing the paper's aligned minimal-rotation
identity; the legacy binary oracle has a nonzero aligned angle drift. The
older reference-frequency XP/XAS ratio above remains evidence for the
explicit binary-compatibility helper, not a paper-derived prediction for the
physical boundary.

This endpoint correction covers zero-nutation rows with at least one exactly
aligned spin. It does not establish a generic two-spin near-aligned limit.
[Chatziioannou et al., Eqs. (21)--(25) and Appendix B](https://arxiv.org/pdf/1703.03967v3)
derive the cubic in \(S^2\), distinguish the two physical turning roots from
the third algebraic root, and order the roots in the elliptic solution.
Their [Appendix E](https://arxiv.org/pdf/1703.03967v3) explains that a PN
expansion can become ill-behaved near spin-orbit alignment even when the
underlying precession frequency is finite, and suggests a small-misalignment
expansion as a possible improvement. Thus a generic small-spin fix requires
more than forcing the aligned waveform when the cubic roots nearly coincide.

The \((36,22)M_\odot\), \((\chi_{1z},\chi_{2z})=(0.35,-0.25)\) fixture is also
an **up-down** system. [Gerosa et al.](https://arxiv.org/abs/1506.09116)
give its physical instability band as
\(r_{\mathrm{ud}\pm}/M=(\sqrt{\chi_1}\pm\sqrt{q\chi_2})^4/(1-q)^2\),
with \(q=m_2/m_1\). The upper edge is \(r/M=6.16079\), or approximately
\(72.865\) Hz using the Newtonian relation
\(r/M=(\pi M f_{22})^{-2/3}\). This estimate is a physical context, not an
XP numerical validity cutoff. On `len`, the generic two-spin
\((\chi_{1x},\chi_{2x})=(10^{-10},-5\times10^{-11})\) waveform differs from
the exact-aligned waveform by about 0.0245 relative L2 over 25--55 Hz,
entirely before that edge. The pre-band discrepancy shows a phase/chart
conditioning problem independently of the physical instability; shrinking
the transverse spins to \(10^{-11}\) does not remove it. Even for both spins
aligned with the orbit, a generic two-spin row has a nonvanishing small-spin
waveform difference over 25--60 Hz, so this is not solely an up-down effect.
As a narrower stable control, with both aligned spin components positive the
two-spin waveform at its 40 Hz reference point approaches its aligned value:
the relative difference is \(2.36\times10^{-11}\) at transverse scale
\(10^{-10}\) and ten times smaller at \(10^{-11}\). Anchoring the phase at
one frequency does not establish broadband continuity.

Across 25--300 Hz the up-down difference rises to about 0.13. An independent
90-digit evaluation of the published cubic for this fixture finds two roots
near \(S^2=0.00977535\) separated by about \(6.89\times10^{-21}\) at
\(\chi_{1x}=10^{-10}\), \(f_{22}=40\) Hz; they are the upper pair there. At
300 Hz the near-coincident pair is the lower pair and the third root is
approximately 0.0282781. This ordering change is inside the predicted
up-down instability region; physical growth of a perturbation and numerical
root/angle-chart effects cannot be separated by an XP/XAS comparison alone.
The two strict expected-failure cases in
`test_xp_two_spin_near_aligned_continuity_pending` record both frequency
regimes. They are not evidence of a generic continuity theorem or a sourced
root-branch repair.

## Executable generic-boundary diagnosis (2026-09-25)

The independent
[`xp_msa_generic_boundary.py`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/xp_msa_generic_boundary.py)
now evaluates the Appendix-B cubic with a private arbitrary-precision context.
It returns the raw depressed-cubic inverse-cosine argument, all three roots,
the unclipped Eq. (23) phase fraction, elliptic parameter, branch-signed
initial phase, and polynomial residuals. It imports neither Torch nor JAX and
does not insert an auxiliary separation. For a coplanar state, Eq. (24)'s
vanishing triple product makes the state an exact turning point; the scalar
uses that identity only to evaluate the endpoint elliptic integral and still
reports the raw quotient. This distinction prevents a working-precision
overshoot from being mistaken for a physical clamp.

For the up-down fixture at 40 Hz, the scalar gives the following derived
conditioning laws as the common transverse scale $x$ tends to zero:

| $x$ | $S_+^2-S_-^2$ | $1+a_{\mathrm{acos}}$ |
|---:|---:|---:|
| $10^{-4}$ | $6.89448368434\times10^{-9}$ | $3.29691982458\times10^{-13}$ |
| $10^{-6}$ | $6.89448597569\times10^{-13}$ | $3.29692523779\times10^{-21}$ |
| $10^{-8}$ | $6.89448597591\times10^{-17}$ | $3.29692523833\times10^{-29}$ |
| $10^{-10}$ | $6.89448597591\times10^{-21}$ | $3.29692523833\times10^{-37}$ |

Thus the physical split is approximately
$0.689448597591491x^2$, while the cubic endpoint distance is approximately
$3296.92523832641x^4$. This is a fixture-specific derivation, not a new MSA
equation. It explains why the endpoint argument becomes unresolvable in
float64 before the turning-root split vanishes. The paired
[`Torch`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/precession_msa_v223_generic_boundary.py)
and
[`JAX`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/precession_msa_v223_generic_boundary.py)
diagnostics expose the raw argument and separately label the clipped
compatibility roots. Their tests require parity where float64 resolves the
cubic and require the unresolved $10^{-10}$ discrepancy to remain visible.
They do not alter waveform dispatch.

The same scalar evaluates Pratten Appendix C Eqs. (67)--(85) with the orbital
angular momentum supplied by the caller. It deliberately reports
$\phi_J$ as undefined when $J_x=J_y=0$. For the tiny generic perturbation,
the chart instead has the finite directional limit
$\alpha_0\to\pi/2+\phi_\mathrm{ref}$, $\phi_J\to0$, while the spin-2
polarization basis remains continuous. A fallback such as
$\phi_J=\pi/2-\phi_\mathrm{ref}$ at exact alignment is therefore a chart
compatibility choice, not an equation supplied by Appendix C. The generic
Torch/JAX projection tests now include source phase $-0.1$, corresponding
to the candidate `phic-phi0` coordinate for `phic=0.3`, `phi0=0.4`. That
cross-frontend result validates the diagnostic source coordinate; it does not
by itself validate replacing the public generic nonzero-`phi0` route. The
subsequent full frame-map and public-binary comparison below closes that
specific azimuth gap.

The published near-alignment literature does not close this gap by supplying
a limiting form of the generic MSA solution. [Klein, Cornish and Yunes
(2013)](https://arxiv.org/abs/1305.1932) construct a separate waveform by
expanding every angular-momentum component in a misalignment parameter
$\epsilon$ (their Eqs. (32), (43)--(45)) and retaining the transverse solution
through $O(\epsilon)$ as two harmonic precession modes. That construction is a
different perturbative model: it does not derive a replacement for the
Chatziioannou $S^2$ turning roots, elliptic initial phase, or the Pratten
Appendix-C source chart. Importing it only at the v223 boundary would therefore
define a new hybrid waveform and matching prescription, not implement a
published continuation of the v223 equations.

The newer $\delta\chi$ formulations resolve a different singular limit.
[Klein (2021), Eqs. (11), (37)--(39)](https://arxiv.org/abs/2106.10291)
reparameterizes the conservative solution so the turning roots remain regular
at equal mass, but its inertial-frame azimuth still contains
$1/\sin^2\theta_L$ and is undefined at exact alignment. [Gerosa et al.
(2023), Eqs. (29)--(34)](https://arxiv.org/abs/2304.04801) likewise identifies
the two smaller $\delta\chi$ roots as the physical turning points and the
largest as spurious, giving a robust root-solving strategy. It also states
that the relative spin azimuth is ill-defined when either spin is aligned and
that a formal Taylor expansion of the zero-amplitude resonance limit remains
future work. These results support a separately implemented precession solver;
they do not provide the missing scalar v223 root/phase/chart continuation.

The papers do not specify a float64 small-misalignment expansion that
preserves the generic MSA branch through the root reordering. Pratten's
minimal-rotation Eq. (7) fixes the exact aligned endpoint, while Chatziioannou
Appendix E identifies the near-aligned PN expansion as ill-conditioned and
suggests a dedicated expansion. Substituting Eq. (51), clipping the cubic
argument, or forcing the aligned chart is therefore not a sourced generic
continuation.

The boundary qualification adopts the narrower behavior that Pratten et al.
state for failed MSA initialization: use the established NNLO prescription.
Production now applies the existing MSA root-validity result at the reference
frequency before selecting the distinct-root clean kernel. Invalid
coalescing-root rows use the inherited NNLO angle route. Generic equal-mass
rows use that same angle fallback. The final-spin option remains independent:
explicit mode 3 raises when its nonaligned MSA averaged-remnant root is invalid,
while explicit mode 0 supplies the continuous NNLO remnant. This introduces
neither a new numerical threshold nor an oracle-fitted coefficient and does
not claim an MSA continuation. A later frame-map derivation routes valid
unequal-mass generic nonzero-`phi0` rows through the clean chart.

### Generic nonzero-azimuth frame map (2026-09-26)

[Pratten et al., Appendix C Eqs. (67)--(68), (76)--(85), Appendix B Eq. (63),
Appendix E Eqs. (110)--(113), and Appendix F Table IV](https://arxiv.org/html/2004.06503v2)
specify the source observer, J frame, Euler/polarization geometry,
quadrupole twist-up, and convention phases. We derive, rather than quote,
the mapping from diffGW's additional observer azimuth to the public binary:
rotate both in-plane spins by `Rz(-phi0)`, use binary `phiRef=phic`, then
multiply both completed quadrupole polarizations by `exp(-2i phi0)` for
conventions 0/1/6. The mpmath scalar Appendix-C chart verifies the rotation;
Torch and JAX complete polarizations verify covariance, and the installed
public LAL binary verifies the mapped full waveform. No LAL implementation
or calibration coefficient enters production.

For the declared 35/25-Msun, two-spin, `phi0=0.4`, v223/mode-3 fixture on
20--512 Hz at 0.25 Hz, direct unshifted plus/cross complex relative L2
against LALSuite 7.26.15 is at most `1.145e-13 / 1.125e-13` across
conventions 0/1/6; flat mismatch is at most `1e-15`. The unchanged strict
gates pass. This closes the scoped generic-azimuth binary comparison; it
does not repair the equal-mass MSA-remnant or near-aligned v223 binary gaps.

For the up-down fixture above, the corrected Torch/JAX waveform residuals are
below $3.21\times10^{-14}$ relative L2 across both tested bands, while the
tiny-spin waveform is below $4.35\times10^{-11}$ from exact alignment. The
former expected-failure continuity cases are therefore strict passing tests
at the unchanged $10^{-6}$ threshold. These are internal paper-route and
cross-frontend results, not external release qualification: the installed LAL
binary follows a different boundary behavior and fails the strict full-complex
waveform gate, as recorded by `scripts/qualify_xp_xphm_boundaries.py`.

The executable gates are in
[`test_xp_v223_generic_boundary_math_reference.py`](https://github.com/xangma/diffgw/blob/main/tests/test_xp_v223_generic_boundary_math_reference.py).
They were run on host `len` under Python 3.12.3 with NumPy 2.5.3, mpmath
1.3.0, SciPy 1.18.1, Torch 2.14.0+cpu, JAX/JAXLIB 0.11.2, pytest 9.1.1,
LALSuite 7.26.15 (LAL 7.7.1, LALSimulation 6.2.1). The LAL packages were
used only as pinned black-box waveform oracles in existing tests; no LAL
source was consulted for this derivation.
