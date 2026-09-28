# Higher-mode SPA convention reference

## Scope and status

`imrphenomhm-spa-conventions/v1`, component
`hm.conventions.quadratic-chirp-spa.v1`, is an independent, reference-only
diagnostic of a quadratic chirp and its phase conventions. It provides no
production change, waveform catalog entry, NR calibration, or physical
qualification. Numerical execution results belong in a separate receipt.

## Inputs, conventions, and options

Let $m$ be a positive integer, $A>0$, $k>0$, $\epsilon>0$, and
$\phi(t)=\phi_0+\omega t+kt^2/2$, with real phase parameters. Define

$$h_{\ell m}(t)=C_{\ell m}A e^{-im\phi(t)}e^{-\epsilon t^2/2},
\qquad P_{\ell m}(f)=\int_{-\infty}^{\infty}\overline{h_{\ell m}(t)}
e^{-2\pi i f t}\,dt,\quad f>0.$$

$C_{\ell m}$ is an independently supplied complex constant. Zero coefficients
give a zero transform but have no unit phase. The optional paper coefficient
phases below assume a positive unequal-mass difference and positive real
leading amplitude factors; odd-mode phases are undefined when their amplitude
vanishes at equal mass. A required `common_td_sign` declares the common TD
strain sign. No phase or sign is inferred from a waveform comparison.

## Mathematics and components

### Exact Gaussian and SPA limit

Conjugating $h_{\ell m}$ gives $+im\phi_0$. Put
$a=\epsilon-imk$ and $d=m\omega-2\pi f$. Completing the square gives

$$-\frac a2t^2+idt+im\phi_0
=-\frac a2\left(t-\frac{id}{a}\right)^2-\frac{d^2}{2a}+im\phi_0.$$

The convergent Gaussian integral therefore yields

$$P_{\ell m}(f)=\overline{C_{\ell m}} A
 e^{im\phi_0}\sqrt{\frac{2\pi}{\epsilon-imk}}
 \exp\left[-\frac{(m\omega-2\pi f)^2}{2(\epsilon-imk)}\right].$$

The principal square root approaches
$e^{i\pi/4}\sqrt{2\pi/(mk)}$ as $\epsilon\to0^+$.
The stationary point is $t_*=(2\pi f/m-\omega)/k$, so

$$P^{\rm SPA}_{\ell m}=\overline{C_{\ell m}} A
\sqrt{\frac{2\pi}{mk}}
e^{i[m\phi(t_*)-2\pi f t_*+\pi/4]}.$$

This limit is exact for the stated oscillatory quadratic-chirp integral.
It does not assert exact SPA for a physical inspiral with varying amplitude
and chirp rate.

### Primitive relative phase and its limitations

For $u=P/|P|$, $g=\gcd(2,m)$, $a_m=2/g$, and $b_m=m/g$, define

$$T_{\ell m}=u_{\ell m}(f)^{a_m}
\overline{u_{22}(2f/m)}^{b_m}.$$

At the matched frequency both chirps have the same stationary point.
Their orbital dynamics cancel because $a_m m=2b_m$:

$$T^{\rm SPA}_{\ell m}
=\left(\frac{\overline{C_{\ell m}}}{|C_{\ell m}|}\right)^{a_m}
\left(\frac{C_{22}}{|C_{22}|}\right)^{b_m}
e^{i(a_m-b_m)\pi/4}.$$

A common orbital shift $u_{\ell m}\mapsto e^{im\alpha}u_{\ell m}$
and time shift $u_{\ell m}(f)\mapsto e^{-2\pi if\tau}u_{\ell m}(f)$
cancel. A common FD strain rotation $e^{i\gamma}$ leaves
$e^{i(2-m)\gamma/g}$; a TD rotation has the conjugate sign.
Thus this is not invariant under an arbitrary common strain rotation.

For $(4,4)$, the primitive powers are $(1,2)$ and a mode sign flip is
observable. The doubled powers $(2,4)$ erase it. For odd $m$, the primitive
mode power is two, so an odd-mode sign is invisible; a factor $i$ is visible.
Simultaneously flipping all odd-$m$ modes equals an orbital shift of $\pi$.
Neither this diagnostic nor a $(2,2)$ phase alignment resolves that ambiguity.

### Supplied leading-PN coefficient phases

[Blanchet et al., arXiv:0802.1249v3](https://arxiv.org/abs/0802.1249v3),
Eqs. (9.3) and (9.4a,b,d,e,h,i), supply these TD unit phases:

| Mode | $C_{\ell m}/|C_{\ell m}|$ before the common TD sign |
| --- | --- |
| $(2,2)$ | $1$ |
| $(2,1)$ | $i$ |
| $(3,3)$ | $-i$ |
| $(3,2)$ | $1$ |
| $(4,4)$ | $-1$ |
| $(4,3)$ | $-i$ |

The transform is derived directly above. Conjugating the entire Fourier-plus
transform of the TD mode gives the carrier defined here, including the
conjugated coefficient and $+\pi/4$ contribution.

Consequently $T_{33}=T_{44}=e^{3\pi i/4}$ for common TD sign $+1$, and
$e^{-\pi i/4}$ for sign $-1$. The declared native London-form offsets
$\Delta_{\ell m}=\pi[3\ell+(\ell+m)\bmod2-2]/2$, entering as
$e^{-i\Delta_{\ell m}}$
([London et al., v4, Eq. 7](https://arxiv.org/pdf/1708.00404v4)), instead give
$T_{33}=T_{44}=-1$.

Choosing the principal angle of $C_{22}$, then removing $m/2$ times the
SPA 22 carrier phase $-\arg C_{22}+\pi/4$, preserves the $(2,2)$ carrier.
The resulting phase correction from
the London-form baseline is, modulo $2\pi$,

$$\delta\Phi_{\ell m}=\begin{cases}
(1-m/2)\pi/4,&\text{common TD sign }+1,\\
-(1-m/2)3\pi/4,&\text{common TD sign }-1.
\end{cases}$$

Changing the $C_{22}$ angle branch by $2\pi$ changes this orbital alignment
by $\pi$, flipping every odd-$m$ carrier. The primitive invariants remain
unchanged. This conditional mathematical difference alone selects neither a
native production convention nor an NR conversion.

For sign $-1$, the multiplier decomposes exactly as
$e^{i(m/2-1)K}=e^{-iK}e^{imK/2}$ with $K=3\pi/4$: a common positive-frequency
carrier phase and an orbital rotation. This identity does not by itself prove a
native defect or a physical polarization-gauge equivalence. Real polarizations
require conjugate factors at negative frequency; multiplication of all positive
frequency carriers by a phase is not automatically a real polarization rotation.
Signed-mode projection and the intended gauge must be checked before production use.
The [signed-pair extension](imrphenomhm-signed-projection.md) derives the
conjugate transport of both Fourier contributions and distinguishes it from
a polarization-basis rotation. Its execution status is recorded separately.

## Implementation mapping

`src/diffgw_reference/imrphenomhm_spa_conventions.py` contains the symbolic
Gaussian/SPA expressions, independent scalar evaluations, primitive invariant,
and optional paper phases. There are no Torch/JAX bindings or production
imports. `tests/test_imrphenomhm_spa_conventions_math_reference.py` maintains
symbolic proofs, direct TD quadrature, damping-limit sequences, gauge tests,
and sign-observability checks.

## Evidence and limitations

The quadrature checks a finite-damping TD integrand against the exact
full-line expression, with a separately bounded omitted Gaussian tail.
The five damping values $10^{-2},10^{-4},10^{-6},10^{-8},10^{-10}$ check
the complex SPA limit, including magnitude, coefficient phase, initial phase,
and square-root branch. All six paper coefficient phases are checked against
the orbital-normalized London-form difference. These checks have no NR input
and do not compare either native waveform backend with a physical inspiral.
The separate [SXS TD/FD convention receipt](../validation/imrphenomhm_spa_conventions_20260928.md)
records the basis conversion, raw-data sign check and fixed-conversion diagnostic.

## Reproduce the checks

On the designated compute host, with the project and optional reference
dependencies installed:

```bash
python -m pytest -q tests/test_imrphenomhm_spa_conventions_math_reference.py
```

The filename joins the maintained strict math-reference inventory
automatically. `--require-math-reference` requires the complete inventory,
so it is not a focused-module flag.
