# Paired spheroidal ringdown projection

The projection preserving `KerrRingdown`'s current convention is

\[
 d_+=\frac{S_-+(-1)^\ell S_+^*}{2},\qquad
 d_\times=\frac{i}{2}\left[S_--(-1)^\ell S_+^*\right],
\]

where the full angular harmonics, including their azimuthal factors, are

\[
 S_+={}_{-2}S_{\ell m}(\theta,\phi;c),\qquad
 S_-={}_{-2}S_{\ell,-m}(\theta,\phi;-c^*),\qquad
 c=a_fM_{\rm sec}(\omega_R-i/\tau).
\]

Here `omega_R` is the physical angular frequency in rad/s, `tau` is in
seconds, and `a_f` is dimensionless. Equivalently, for a dimensionless
author frequency \(\bar\omega=M\omega_R-iM/\tau\), use
\(c=a_f\bar\omega\). The mirror argument is **\(-c^*\)**.

## Primary equations and derivation

The only external input is [Cook & Zalutskiy, arXiv:1410.7698v2](https://arxiv.org/pdf/1410.7698v2):
Eq. (3) uses \(e^{-i\Omega t+im\phi}\), Eq. (4) has \(c=a\Omega\),
Eqs. (48b,c) give the angular reflection and conjugation symmetries,
and Eq. (57) gives the mirror frequency \(-\Omega^*\).
The angular phase is the real-positive coefficient at spherical index
\(j=\ell\), consistent with the normalization following Eq. (56).
The primary paper's local text was read directly; no external numerical
implementation, oracle output, or fitted correction was used.

Set \(p=(-1)^\ell\) and \(\Omega=\omega_R-i\gamma\), with
\(\gamma=1/\tau>0\). Including the \(e^{im\phi}\) factor, Eqs. (48b,c)
give

\[
 {}_{-2}S_{\ell,-m}(\theta,\phi;-c^*)
 =p\left[{}_{-2}S_{\ell m}(\pi-\theta,\phi;c)\right]^*.
 \tag{1}
\]

For complex strain \(H=h_+-ih_\times\), write the paired modes as

\[
 H=B S_+e^{-i\Omega u}+D S_-e^{+i\Omega^*u},\qquad u=t-t_0.
\]

Equatorial reflection, \(H(\pi-\theta,\phi)=H(\theta,\phi)^*\),
together with (1), requires \(D=pB^*\). The repository computes
\(h_{+,\times}=\operatorname{Re}[q\,d_{+,\times}]\), with
\(q=Ae^{-\gamma u}e^{+i\omega_Ru}\), where \(A\) includes the supplied
phase. Its existing spherical coefficients fix
\(D=A/2\) and \(B=pA^*/2\), so

\[
 H=\frac12\left[q S_-+p q^*S_+\right]. \tag{2}
\]

Taking the real part and minus the imaginary part of (2) yields the
projection at the top of this note. This also proves the cross-polarization
sign and preserves the existing factor-of-two amplitude convention.

## One eigensolve and the spherical limit

Equation (1) permits evaluation using only the \(+m,c\) eigensolution:

\[
 d_+=\frac p2\left[S_+(\pi-\theta,\phi)^*+S_+(\theta,\phi)^*\right],
\quad
 d_\times=\frac{ip}{2}\left[S_+(\pi-\theta,\phi)^*-S_+(\theta,\phi)^*\right].
\]

Alternatively, spherical expansion coefficients obey
\(C_{j,\ell,-m}(-c^*)=(-1)^{j+\ell}C_{j,\ell,m}(c)^*\).
The positive anchor is preserved because the sign is unity at \(j=\ell\).

At \(c=0\), \(S_+={}_{-2}Y_{\ell m}\) and
\(S_-={}_{-2}Y_{\ell,-m}\). Substitution gives exactly
`_swsh_projection_coefficients`, including odd \(\ell\), arbitrary
azimuth, and its \(+i/2\) cross coefficient. Equation (1) also gives
\(d_\times=0\) at the equator.

A float64 check of all six currently supported spherical mode pairs,
six inclinations including both poles and the equator, and nonzero azimuth
found maximum absolute differences of \(1.48\times10^{-16}\) in the
spherical limit, \(3.65\times10^{-16}\) in (1) at
\(c=0.52-0.087i\), and \(3.11\times10^{-17}\) between (2) and the
real-polarization reconstruction.

There is no remaining angular conjugation ambiguity under these conventions.
The amplitude dictionary retains its existing meaning: its positive-time-
exponential coefficient corresponds to twice the mirror-mode strain
amplitude. Independent mirror excitation amplitudes would be a different
API from the equatorially symmetric pairing derived here.
