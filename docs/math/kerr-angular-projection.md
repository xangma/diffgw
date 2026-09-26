# Kerr spin-weighted angular projection

Specification ID: `kerr-angular-projection-scalar/v1`.

## Scope and status

This specification checks the spin-weight $s=-2$ spherical and spheroidal
angular factors used to project positive-$m$ Kerr ringdown modes into complex
plus and cross coefficients. Its independent scalar reference covers six
spherical modes and two spheroidal modes at stated complex oblateness values.
It does not qualify QNM frequencies, excitation amplitudes, a full ringdown
waveform, all supported modes or the angular-series stopping rule.

## Inputs, conventions, and options

Let $c=a\omega$ be the complex angular oblateness, $\theta$ the inclination,
and $\phi$ the azimuth. The spherical basis has unit sphere norm and
$e^{im\phi}$ dependence. A spheroidal eigenfunction has unit Hermitian norm
in that basis; its coefficient at the requested $\ell$ is real and positive.
The reference truncates at $\ell_{\max}=10$ and integrates matrix elements
with 32-point Gauss--Legendre quadrature. The production solvers use their
own spectral matrix and default $\ell_{\max}=18$.

## Mathematics and components

Cook and Zalutskiy, [arXiv:1410.7698v2](https://arxiv.org/abs/1410.7698),
Eqs. (4), (48), and (49)--(57), specify the spin-weighted angular equation,
spherical expansion, and reflected negative-$m$ partner. For $s=-2$, write
$x=\cos\theta$ and ${}_{-2}Y_{jm}$ for the spherical basis. The scalar
reference obtains each basis function from SymPy's Wigner-$d$ matrix, then
integrates the multiplication operators directly:

$$
X_{jk}=\int {}_{-2}Y_{jm}^{*}\,x\,{}_{-2}Y_{km}\,d\Omega,
\qquad
X^{(2)}_{jk}=\int {}_{-2}Y_{jm}^{*}\,x^2\,{}_{-2}Y_{km}\,d\Omega.
$$

It diagonalizes the finite angular operator

$$
H_{jk}(c)=[j(j+1)-2]\delta_{jk}-4cX_{jk}-c^2X^{(2)}_{jk},
$$

continuing the eigenvalue from $A_{\ell m}(0)=\ell(\ell+1)-2$ along a straight
path to $c$. If $S_{\ell m}(\theta,\phi;c)$ is the resulting normalized
eigenfunction, the projected coefficients are

$$
P=\tfrac12[S_{\ell m}(\theta,\phi;c)
             +S_{\ell m}(\pi-\theta,\phi;c)],\qquad
C=-\tfrac{i}{2}[S_{\ell m}(\theta,\phi;c)
             -S_{\ell m}(\pi-\theta,\phi;c)].
$$

At $c=0$, the spherical check independently constructs the partner as
$(-1)^\ell{}_{-2}Y_{\ell,-m}^{*}(\theta,\phi)$. The reflected form above
follows from the equatorial partner symmetry in the cited equations.

## Implementation mapping

`diffgw_reference.kerr_angular` uses SymPy Wigner-$d$, mpmath quadrature and
mpmath eigensolutions. It imports no production angular helper or matrix.
`tests/test_kerr_angular_math_reference.py` compares its complex $P,C$ and
their $\theta,\phi$ derivatives with the Torch and JAX
`_ringdown_projection` paths. It separately checks angular eigenvalues and
the Torch derivative with respect to $\operatorname{Re}c$.

## Evidence and limitations

The CPU FP64 gates were declared before evaluating the reference: absolute
errors below $3\times10^{-13}$ for spherical values, $2\times10^{-12}$ for
their angle derivatives, $3\times10^{-10}$ for spheroidal values and
eigenvalues, $2\times10^{-9}$ for their angle derivatives, and $3\times10^{-6}$
for the Torch real-$c$ derivative against a scalar centered difference.
The spherical grid covers $(\ell,m)=(2,2),(2,1),(3,3),(3,2),(4,4),(4,3)$
at three off-axis $(\theta,\phi)$ pairs. Spheroidal cases are $(2,2)$ at
$c=0.42-0.10i$ and $(3,2)$ at $c=0.58-0.12i$.

The 42 checks pass on len with Torch 2.14.0+cpu, JAX 0.11.2, SymPy 1.14.0,
and mpmath 1.3.0. The JAX spheroidal eigenvector currently comes from a
`pure_callback` and has no derivative with respect to $c$; only its angle
derivatives are qualified. Neither this component gate nor the separate
public-binary waveform gate establishes a full parameter-domain result.

## Reproduce the checks

Run `PYTHONPATH=src:tests python -m pytest -q
tests/test_kerr_angular_math_reference.py --reference-device=cpu` in an
environment with the required scalar and frontend packages. The full strict
math gate collects all `test_*math_reference*.py` modules and uses
`--require-math-reference`.
