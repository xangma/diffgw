# IMRPhenomD composed polarizations

Version: **`imrphenomd-composed-polarizations/v1`**. The scalar
[`imrphenomd_waveform.py`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/imrphenomd_waveform.py)
composes the unchanged [amplitude](imrphenomd-amplitude.md) and
[phase](imrphenomd-phase.md) specifications with the audited
[unit conversions](units.md). It imports no production waveform code or
coefficients. Its default fits retain only the digits printed in
[Khan et al., arXiv:1508.07253v2](https://arxiv.org/html/1508.07253v2),
Appendix C, Table 5. Tests pass the frontend's inherited extra digits as
explicit inputs; agreement conditional on those rows does not prove their
publication provenance. The dimensionless QNM pair $(f_{\rm RD},f_{\rm damp})$
is also an explicit input, so this slice does not certify the remnant or QNM
pipeline.

For positive physical frequency $f$ in hertz, component masses in nominal
solar masses, and luminosity distance $d$ in megaparsecs, define
$M_s=(m_1+m_2)T_\odot^{\rm N}$, $D_s=dL_{\rm Mpc}/c$, and $u=M_sf$. The
composed amplitude is

$$
\mathcal A(f)=\sqrt{\frac{5\eta}{24}}\,
\frac{M_s^{5/6}}{\pi^{2/3}D_s f^{7/6}}\,a(u),
$$

where $a(u)$ is the dimensionless, piecewise amplitude from the amplitude
specification. The physical stationary-phase scale and inclination factors
follow [Cutler and Flanagan, arXiv:gr-qc/9402014v1](https://arxiv.org/pdf/gr-qc/9402014v1),
Eqs. (62)–(63). Khan et al. Eqs. (17)–(18) define the normalization used when
fitting $a(u)$ in their own dimensionless convention; the physical scale above
is the explicit DiffGW adapter. The nominal solar mass, megaparsec and speed
of light definitions are traced in the unit specification.

Let $\phi(u)$ be the joined intrinsic phase, and $s_{\rm peak}$ the
[frontend time-gauge slope](imrphenomd-phase.md) using the raw fitted
$\alpha_1$ at the amplitude peak. DiffGW's phase conventions are

$$
\begin{aligned}
\Psi_{\rm coal}(f)&=-\pi/4+\phi(u)-s_{\rm peak}u
                       +2\pi f t_c-2\phi_c,\\
\Psi_{\rm ref}(f)&=\phi(u)-\phi(u_r)-s_{\rm peak}(u-u_r)
                       +2\pi f t_c-2\phi_c,\quad u_r=M_s f_{\rm ref}.
\end{aligned}
$$

The reference-frequency subtraction, phase offsets, and time gauge are
project conventions; the paper defines the intrinsic amplitude and phase,
not these API choices. With inclination $\iota$ in radians, the returned
complex positive-frequency polarizations are

$$
\tilde h_+=\frac{1+\cos^2\iota}{2}\mathcal A e^{-i\Psi},\qquad
\tilde h_\times=-i\cos\iota\,\mathcal A e^{-i\Psi}.
$$

The negative exponent follows Khan et al. Eq. (7). The signed cross quadrature
and choice of orbital-phase origin fix DiffGW's polarization convention. They
must be kept fixed when comparing complex strains. The scalar reference
returns zero strain above the inherited $u=0.2$ default cutoff; this cutoff is
a frontend choice, not a fitted paper coefficient. The present reference
requires positive frequencies and covers the untapered default waveform.
Additional Torch-only tapers are outside its scope.

## Smooth-branch derivatives

The reference derives complex-polarization slopes from the same sourced
amplitude and phase, with fit rows and the dimensionless QNM pair held fixed.
Write $H=\mathcal A e^{-i\Psi}$, let $a'=da/du$ and $\phi'=d\phi/du$ on the
selected branch, and hold $f_{\rm ref}$ fixed when present. The product rule gives

$$
\frac{\partial H}{\partial f}=\mathcal A e^{-i\Psi}\left[
-\frac{7}{6f}+M_s\frac{a'}{a}
-i\left(M_s(\phi'-s_{\rm peak})+2\pi t_c\right)\right].
$$

The executable expression avoids division by $a$. The polarization factors
multiply this derivative unchanged. For a common rescaling
$(m_1,m_2)\mapsto(\lambda m_1,\lambda m_2)$ at $\lambda=1$, fixed spins,
physical frequencies and distance, define

$$
P=u(\phi'(u)-s_{\rm peak})-
\begin{cases}
0,&\text{coalescence convention},\\
u_r(\phi'(u_r)-s_{\rm peak}),&\text{reference-frequency convention}.
\end{cases}
\qquad
\frac{\partial H}{\partial\ln\lambda}
=\mathcal A e^{-i\Psi}\left(\frac56+u\frac{a'}{a}-iP\right).
$$

The amplitude exponent follows the physical stationary-phase normalization;
the rest follows $u=M_sf$ and the phase adapter above. Analytic extrinsic
identities are $\partial_d\tilde h=-\tilde h/d$,
$\partial_{t_c}\tilde h=-2\pi if\tilde h$ and
$\partial_{\phi_c}\tilde h=2i\tilde h$. Inclination derivatives follow the
two explicit polarization factors. The tests compare real and imaginary
Torch/JAX autodiff Jacobians with these scalar expressions at interior points
of all three frequency regions. Frequency and mass-scale derivatives at the
hard cutoff are undefined; the tests avoid all branch boundaries. The fixed QNM
pair excludes remnant and QNM response to changing mass ratio or spins from the
mass-scaling check.

## Component traceability and checks

| Component ID | Evidence | Reference function | Torch/JAX check |
| --- | --- | --- | --- |
| `phenomd.waveform.physical_scale.v1` | Cutler–Flanagan Eqs. (62)–(63); [unit definitions](units.md) | `physical_scale` | Direct absolute complex strain, without amplitude rescaling |
| `phenomd.waveform.phase_adapter.v1` | Khan Eq. (7); [DiffGW phase conventions](imrphenomd-phase.md) | `phase_adapter` | Both phase conventions, nonzero $t_c$ and $\phi_c$ |
| `phenomd.waveform.polarizations.v1` | Khan Eqs. (5)–(7); Cutler–Flanagan Eq. (62); project cross sign | `polarizations` | Both complex polarizations and inclination identities |
| `phenomd.waveform.default_cutoff.v1` | DiffGW frontend choice | `default_cutoff` | Zero strain beyond $u=0.2$ |
| `phenomd.waveform.polarization_derivatives.v1` | Product/chain rules applied to the sourced expressions and project conventions | `polarization_derivatives` | Frequency, common-mass and extrinsic complex Jacobians against both frontends |

`COMPONENT_BINDINGS` resolves every ID to the scalar function, Torch and JAX
entry points, and a maintained test. The FP64 cases include equal and unequal
masses, aligned spins, both sides of the amplitude and phase joins, a supplied
QNM pair, merger-ringdown, and a point beyond cutoff. Comparisons use direct
complex error and unwrapped phase without optimizing time or phase. The test
also checks the edge-on/face-on polarization identities and inverse-distance
scaling analytically. The selected cases are implementation verification, not a
uniform error bound or independent physical validation. A separate
[intrinsic composition](imrphenomd-intrinsic.md) now checks selected mass-ratio
and spin derivatives with the remnant and QNM chain included. FP32, full-domain
intrinsic derivatives, batch/JIT transforms, QNM endpoints and Torch's opt-in
tapers remain further qualification work.

Run `python -m pytest -q tests/test_imrphenomd_waveform_math_reference.py` for
this slice. The full strict mathematical-reference gate also collects it.
