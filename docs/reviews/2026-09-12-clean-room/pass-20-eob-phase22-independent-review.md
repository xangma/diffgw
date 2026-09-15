# Pass 20 independent review: SEOBNRv4 h22 phase

## Decision

**RETAIN the no-patch blocker.** The native phase building block has no
source-complete correction that can safely be applied from the permitted
papers. Its factorized phase, tail-radius conversion, A16 residual, and
explicit choice plumbing are supported. The Bohé energy symbol is internally
inconsistent with the paper's own PN-order and test-particle statements, and
the NQC calibration is not published as a complete arbitrary-parameter rule.
Neither issue authorizes silently changing the runtime.

This is an independent review of the primary report, performed after reading
the source equations. I did not inspect implementation ports, generated code,
notebooks, git history, verification outputs, binaries, or residuals. No
runtime or test file was edited.

## Primary-source ledger

The local TeX files below are the extracted `paper.tex` assets downloaded from
the official arXiv source URLs. Hashes are SHA-256 of that local file.

| source | version and URL | local asset | SHA-256 |
| --- | --- | --- | --- |
| Damour, Iyer, Nagar, factorized modes | [arXiv:0811.2069v2](https://export.arxiv.org/e-print/0811.2069v2) | `/tmp/pass20phase/din/paper.tex` | `768f0daaabd403e8908388166ea575900b8948463b201ac0e2d94473701da146` |
| Pan et al., spinning factorized modes | [arXiv:1006.0431v2](https://export.arxiv.org/e-print/1006.0431v2) | `/tmp/pass20phase/pan/paper.tex` | `ba57ba0a5a5ada831696dc8e7a47a1e003629087d23ce7dc7531ae7e7bc6d441` |
| Taracchini et al., spinning EOB | [arXiv:1202.0790v2](https://export.arxiv.org/e-print/1202.0790v2) | `/tmp/pass20phase/tar12/paper.tex` | `79ba4c7c3ccd6fe180141f36714693d992946804f468056e3000622fa28e7297` |
| Taracchini et al., generic-mass-ratio spinning EOB | [arXiv:1311.2544v1](https://arxiv.org/abs/1311.2544v1) | reviewed PDF asset | `f57b099e701c1206802d9887f6843ce75fc8753ac9c3ae5c2137933160f85967` |
| Bohé et al., SEOBNRv4 | [arXiv:1611.03703v3](https://export.arxiv.org/e-print/1611.03703v3) | `/tmp/pass20phase/seob4/paper.tex` | `08d6f6005b7397f753fa6956edcee6ba96052e0ded683827db5f66793d5917c1` |

## Independent derivation

Taracchini Eq. (hlm) and Bohé Sec. II B use

\[
h^{F}_{\ell m}=h^{(N,\epsilon)}_{\ell m}S^{(\epsilon)}_{\rm eff}
T_{\ell m}e^{i\delta_{\ell m}}(\rho_{\ell m})^\ell,
\qquad h^{\rm insp-plunge}_{\ell m}=h^F_{\ell m}N_{\ell m}.
\]

For 22, Taracchini Appendix A gives a negative real Newtonian prefactor.
Therefore, with the native convention `h22 = A exp(-i phase22)`, continuous
orbital phase \(\Phi\) gives

\[
\mathrm{phase}_{22}=2\Phi-\arg T_{22}-\delta_{22}+\pi.
\]

This verifies the native `+pi` and the use of a continuous orbital phase. It
does not justify unwrapping a sampled complex argument or replacing the
orbital phase by a frequency integral in this building block.

The tail equation in DIN Eq. (19), Pan Eq. (26), and Taracchini Appendix A
uses \(\widehat k=m\Omega H\) and
\(\exp[2i\widehat k\log(2m\Omega r_0)]\). Taracchini specifies
\(r_0=2M/\sqrt e\). The shared native phase uses the DIN radius \(2M\); for
22, changing the radius multiplies the phase by
\(2\widehat k\log(1/\sqrt e)=-\widehat k=-2\omega(H/M)\).
Thus `tail22_phase = _tail_phase(2,2,omega,energy)-2*omega*energy` is the
exact radius conversion under the total-energy convention used by DIN's tail
definition. The conversion itself is not an empirical fit.

Taracchini Eq. (A16) independently gives

\[
\delta_{22}=\frac73u+\frac{428\pi}{105}u^2
 +\left(\frac{1712\pi^2}{315}-\frac{2203}{81}\right)u^3
 -24\nu\omega^{5/3},\qquad u=\omega H/M.
\]

This matches `delta22_nonspin` exactly, including the separate \(-24\nu
\omega^{5/3}\) term.

## Bohé spin term: unresolved source notation

Bohé Eq. (2.6) prints

\[
-\frac43[\chi_S(1-2\nu)+\chi_A\delta](\Omega H_{\rm EOB})^2,
\]

with the bracket equal to the Kerr-spin combination used by the native
`q_kerr`. However, Bohé Eq. (2.1) defines \(H_{\rm EOB}=M\sqrt{1+2\nu
(H_{\rm eff}/\mu-1)}-M\), i.e. the binding-energy normalization. Taken
literally, that makes this displayed correction begin at \(v^{10}\), not the
stated 3PN \(v^6\), and it vanishes in the test-particle limit. DIN instead
defines the real Hamiltonian used in its tail as the total ADM energy and
explicitly obtains \(H_{\rm real}\to M\) as \(\nu\to0\).

The paper supplies no erratum or separate rule selecting which normalization
Eq. (2.6) intended. The native function's total-energy use reproduces the
stated PN order and Kerr limit, but its own docstring correctly labels that
choice a declared inference. Replacing it with literal binding energy would
be equally unsupported and would contradict those stated limits. This is a
source blocker, not an exact safe patch.

Pan Eq. (27a)'s \(-4qv^6/3+20qv^8/63\) terms are test-particle Kerr terms;
Pan does not prescribe the comparable-mass substitution \(q\mapsto q_{\rm
Kerr}\) for SEOBNRv4. The earlier positive-spin Taracchini calibration
(`540 nu max(chi,0) omega^2`) is likewise not declared by Bohé as an
additional v4 term. Keeping both as explicit caller choices is therefore
source-faithful; selecting a default sum is not.

## NQC and native comparison

Bohé's NQC equation (2.7) has three real amplitude coefficients
\(a_{1,2,3}^{h_{22}}\) and two real phase coefficients
\(b_{1,2}^{h_{22}}\). They are determined by matching amplitude, amplitude
curvature, GW frequency, and GW-frequency slope to NR input fits at the
calibrated peak offset. The paper gives neither a complete machine-readable
parameter table nor an analytic arbitrary-parameter coefficient prescription.
Consequently, a complete SEOBNRv4 inspiral-plunge mode cannot be reconstructed
from the permitted source assets. The native `seobnrv4_published_mode22`
explicitly returns the pre-NQC factorized mode, which is an honest bounded
scope.

After derivation, the native comparison is:

* `tail22_phase` implements the exact radius conversion above;
* `delta22_nonspin` implements A16;
* `delta22_bohe_spin` exposes the unresolved total-versus-binding convention
  rather than hiding it;
* `seobnrv4_published_phase22` implements the factorized phase gauge and
  constant \(\pi\);
* the residual composer requires all unresolved inheritance choices
  explicitly, and the mode assembler documents its pre-NQC boundary.

The direct source-derived tests cover the tail special-function phase and
derivatives, Newtonian sign, A16 high-order term, the Bohé PN-order/Kerr-limit
inference, explicit choice independence, batching, exchange symmetry, and
gradients. No public-binary or parity test was run. The no-patch decision is
**RETAIN**: the exact safe correction is none until the Bohé energy notation
and NQC calibration metadata are resolved by an authoritative source.
