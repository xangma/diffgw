# Pass 20: SEOBNRv4 (2,2) phase, primary-source audit

## Scope and method

This is a clean-room audit of the published SEOBNRv4 inspiral-plunge h22
phase. I used the original papers listed below and inspected the native
TorchWave phase module only after deriving the source equations. I did not
use LALSuite, PyCBC, TEOB, translated or generated implementations, author
notebooks, git history, public binaries, residuals, or verification outputs.
No archived mathematical asset contains the SEOBNRv4 h22 NQC coefficients,
so no .m asset was used for this pass.

## Source ledger

The recorded hashes are SHA-256 hashes of the downloaded arXiv PDF at the
version in the URL.

| source | version and URL | SHA-256 |
| --- | --- | --- |
| Damour, Iyer, Nagar, factorized modes | [arXiv:0811.2069v2](https://arxiv.org/abs/0811.2069v2) | a60049d73b78f21fd0421ada236f9635c5023de8ab28a489aa3c1bd78857ba21 |
| Pan et al., spinning factorized modes | [arXiv:1006.0431v2](https://arxiv.org/abs/1006.0431v2) | 4b51940aae4299c129b08d64f7f5ae92a5f9de00e96f2653abe719e76d596c19 |
| Taracchini et al., prototype spinning EOB | [arXiv:1202.0790v2](https://arxiv.org/abs/1202.0790v2) | d6d126be90bc789835ec363451f4ec51728a8d6ddc09eff078620fdc46860745 |
| Taracchini et al., generic-mass-ratio spinning EOB | [arXiv:1311.2544v1](https://arxiv.org/abs/1311.2544v1) | f57b099e701c1206802d9887f6843ce75fc8753ac9c3ae5c2137933160f85967 |
| Bohé et al., SEOBNRv4 | [arXiv:1611.03703v3](https://arxiv.org/abs/1611.03703v3) | db9dc054368f131bdc152b319e54441c665547668221891723b309b005ac23ba |

## Published factorization and phase convention

The factorized mode is stated by Damour et al. (Eq. (1)) and repeated by
Pan et al. (Eq. (2)) and Bohé et al. (Sec. II B):

    h_lm^F = h_lm^(N,epsilon) S_hat_eff^(epsilon) T_lm
             exp(i delta_lm) (rho_lm)^l .

For the even-parity 22 mode, the source factor is
S_hat_eff = H_eff/mu (Pan Eq. (25); Taracchini et al. 2012 Eq. (A5)).
The orbital velocity entering the Newtonian and residual amplitude factors
is the EOB replacement v_Phi = r_Omega Omega_hat, with
r_Omega = Omega_hat (partial H_hat_real / partial p_Phi at p_r=0)^(-2/3)
(Taracchini 2012 Eq. (A2)); it is not an independently guessed Keplerian
frequency for the phase.

The Newtonian 22 prefactor carries a minus sign in the source convention
(Taracchini 2012 Eqs. (A1), (A3), and Pan Eq. (38a)). Thus, when the complex
mode is represented as h22 = A exp(-i phase22), a continuous orbital phase
Phi gives

    phase22 = 2 Phi - arg(T22) - delta22 + pi .

The papers do not define a sampled-argument unwrap or an alternate constant
gauge; the +pi is the direct consequence of the printed Newtonian sign.

## Tail factor and radius convention

Damour et al. Eq. (19) and Pan et al. Eq. (26) give

    T_lm = Gamma(l+1 - 2 i m Omega H_real) / Gamma(l+1)
           exp(pi m Omega H_real)
           exp(2 i m Omega H_real log(2 m Omega r_0)).

Taracchini 2012 Eq. (A6) specifies r_0 = 2M/sqrt(e), with
Omega_hat = M Omega and H_real in the real-energy normalization used by
that equation. For 22, k_hat = 2 omega (H_real/M) and the logarithm is
log(8 omega/sqrt(e)). Relative to the common r_0 = 2M form, this changes
the continuous phase by -k_hat. This is an exact radius conversion, not an
empirical correction.

## Residual phase terms

Taracchini 2012 Eq. (A16) prints the nonspinning residual phase completely:

    delta22_A16 = 7u/3 + (428 pi/105)u^2
                 + (1712 pi^2/315 - 2203/81)u^3
                 - 24 nu omega^(5/3),
    u = omega H_real/M .

Bohé et al. Eq. (2.6) adds the SEOBNRv4 spin phase correction

    delta22_spin,v4 = -(4/3) [chi_S (1-2 nu) + chi_A delta]
                     (Omega H_EOB)^2 .

The bracket is equivalently the Kerr-limit spin combination
q_Kerr = (1-2 nu) chi_S + delta chi_A. The paper prints H_EOB in this
term; it does not publish a separate implementation prescription that
replaces it by binding energy.

There are two additional source-era terms that cannot be selected uniquely
for SEOBNRv4 from the allowed papers:

* Pan et al. Eq. (27a) gives test-particle Kerr terms, including the
  -4 q_Kerr v^6/3 term and a higher-order 20 q_Kerr v^8/63 term. Pan's
  equation does not establish that replacing the Kerr spin by q_Kerr gives
  a comparable-mass SEOBNRv4 prescription.
* Taracchini et al. (1311.2544, text following Eq. (4)) describe the
  positive-spin calibration +540 nu chi (M Omega)^2, with
  chi = chi_S + chi_A delta/(1-2 nu). Bohé's SEOBNRv4 Sec. II B introduces
  its Eq. (2.6) correction but does not say to retain this earlier calibrated
  term.

Consequently, the primary record supports the A16 term plus the explicit
Bohé Eq. (2.6) term, but does not support one unique sum containing Pan's
Kerr terms and/or the older 540 calibration. The native module therefore
requires those inheritance choices explicitly and does not silently claim a
unique v4 residual phase.

## SEOBNRv4 NQC requirement and blocker

Bohé et al. Eq. (2.7) defines the inspiral-plunge correction
h_lm^(insp-plunge) = h_lm^F N_lm. For 22, N_22 contains three real
amplitude coefficients and two phase coefficients multiplying powers of
p_rstar, r, and r Omega_hat. The paper states that these coefficients are
fixed by matching amplitude, frequency, and their derivatives to numerical
relativity waveforms. The paper also describes the broader NR calibration,
but does not publish a complete machine-readable table or an analytic
formula for the coefficients needed at arbitrary parameters.

The native seobnrv4_mode22.py is consequently only the pre-NQC factorized
building block. Completing the published inspiral-plunge mode would require
NQC fitted data and trajectory quantities owned by other modules, which are
outside this pass. Implementing NQC or inventing coefficients would not be
source-complete.

## Native comparison and decision

After the derivation, I inspected
src/torchwave/approximants/eob/seobnrv4_mode22.py:

* tail22_phase (lines 55--69) uses the shared r0=2M tail phase and
  subtracts 2*omega*hreal_over_M, which is the exact
  2M to 2M/sqrt(e) conversion above.
* delta22_nonspin (lines 79--90) is the printed Taracchini A16 expression.
* delta22_bohe_spin (lines 93--103) is the Bohé Eq. (2.6) bracket with
  q_kerr; its docstring correctly identifies the total-energy choice as a
  declared inference because the paper's notation is not an API contract.
* seobnrv4_published_phase22 (lines 190--204) implements the factorized
  phase convention 2 Phi - arg(T) - delta + pi.
* seobnrv4_published_mode22 (lines 207--235) explicitly documents that it
  returns the factorized mode before NQC.

No exact, source-complete runtime correction was justified. I made no
runtime or test changes. The remaining blockers are the unselected
inheritance of older optional residual-phase terms and, decisively, the
unpublished parameter-dependent NQC calibration data. The focused native
phase tests already cover the source equations, batching, and gradients;
public-binary verification was not run.

Validation: git diff --check passed for this report.
