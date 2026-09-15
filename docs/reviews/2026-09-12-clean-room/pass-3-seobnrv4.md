# SEOBNRv4 conservative reconstruction, pass 3

The separate `SEOBNRv4Hamiltonian` class implements an aligned-spin conservative
Hamiltonian reconstructed from the primary papers below. It is available for
review through `torchwave.approximants.eob.seobnrv4_hamiltonian` and as the
explicit `SEOBNR_ODE` option `conservative_model="seobnrv4_published"`. The
default remains `"prototype"`. This work does not establish
full SEOBNRv4 waveform or binary compatibility.

Only fresh primary publications and the existing repository's parameter and
derivative interfaces were used. No LAL/PyCBC implementation, implementation
port, binary oracle, residual fit, prior calibration audit, or implementation
coefficient table was consulted. Calibration decimals were transcribed at the
precision printed in the v4 paper. Two inconsistent printed formulas require
the explicitly documented interpretations below.

## Primary source ledger

PDF hashes are SHA-256 of the retrieved bytes. Equation numbers refer to the
PDF, since HTML conversions sometimes renumber them.

| Primary publication | Equations used | PDF SHA-256 |
| --- | --- | --- |
| [Bohé et al., arXiv:1611.03703v3](https://arxiv.org/pdf/1611.03703v3) | Energy map (2.1); radial potential and 4PN addition (2.2)-(2.3); spin variable (2.8); calibration (4.8), (4.12)-(4.14); Sec. II A's specification of the earlier spin map | `db9dc054368f131bdc152b319e54441c665547668221891723b309b005ac23ba` |
| [Barausse and Buonanno, arXiv:0912.3517v2](https://arxiv.org/pdf/0912.3517v2) | Deformed metric and derivatives (5.36), (5.47)-(5.50); leading spin-orbit limit (5.53), (5.56); aligned Hamiltonian (6.1)-(6.5) | `5757f6cd9e1ffd808c64347a7fc4ffb637b63613dd8467aae4a86c3d436bb80a` |
| [Barausse and Buonanno, arXiv:1107.2904v2](https://arxiv.org/pdf/1107.2904v2) | Dynamical spin map (50)-(52), Hamiltonian and spin-squared term (53)-(54) | `1e829ba28d7a2371ede558c3b30b2914414aa50df9ef837167afe708c28fc12e` |
| [Taracchini et al., arXiv:1202.0790v2](https://arxiv.org/pdf/1202.0790v2) | Radial potentials (7)-(10), tortoise transformation and quartic discussion (14)-(15) | `d6d126be90bc789835ec363451f4ec51728a8d6ddc09eff078620fdc46860745` |
| [Taracchini et al., arXiv:1311.2544v1](https://arxiv.org/pdf/1311.2544v1) | Log potential (2)-(3); p. 2's zero-gauge prescription and updated spin-orbit/spin-spin calibration insertions | `f57b099e701c1206802d9887f6843ce75fc8753ac9c3ae5c2137933160f85967` |
| [Steinhoff et al., arXiv:1608.01907v2](https://arxiv.org/pdf/1608.01907v2) | Log-resummed coefficients (A1)-(A2), radial potential (A3), nonspinning tortoise quartic (A5) | `f338af0f07b455f4e8f02f830a278e9ede36cfa4c8c0208e7df685c7912d07c6` |
| [Damour and Nagar, arXiv:0704.3550v2](https://arxiv.org/pdf/0704.3550v2) | Nonspinning tortoise Hamiltonian (6)-(7), Hamilton equations (8)-(11) | `9c20b4d0835c1ea4eccd65b1677f65a1054371570ed02de0b6ac5a6c473af352` |

The [2012 erratum to the spinning-test-particle Hamiltonian](https://journals.aps.org/prd/pdf/10.1103/PhysRevD.85.069904)
was also checked: its correction concerns a coefficient in the ADM 3.5PN
expansion, Eq. (6.20), rather than the curved-spacetime Hamiltonian used here.

## Normalization and implementation

The class uses total mass `M=1`, momenta divided by reduced mass `mu`, and spins
divided by `M^2`. With mass fractions `X1`, `X2`, its two spin combinations are

```text
a = S_Kerr/M^2 = chi1 X1^2 + chi2 X2^2
sigma_star/M^2 = nu (chi1 + chi2)
```

The effective particle's spin map is the zero-gauge specialization of
Barausse–Buonanno (2011), Eqs. (51)-(52). It includes the published 2.5PN and
3.5PN terms with `Q-1 = gamma^ij p_i p_j` and
`g^rr p_r^2` as explicitly prescribed below Eq. (52).
Taracchini (2013), p. 2, changes the calibrated 4.5PN insertion to
`dSO nu S_Kerr/r^3` and the extra spin-spin Hamiltonian term to
`dSS nu (S1^2+S2^2)/r^4`. Those prescriptions are retained by Bohé Sec. II A.
The earlier, different v1 insertions in Taracchini (2012) are therefore not used.

`K`, `dSO`, and `dSS` use the literal polynomials and decimal precision of
Bohé Eqs. (4.8), (4.12)-(4.14), evaluated at
`chi = a/(1-2 nu)`. The radial coefficients come from Steinhoff (A2), with the
Bohé (2.3) addition to `Delta5`. Algebraic cancellation of the apparent
`1/nu` in (A2c) makes these coefficients regular at `nu=0`.

The equatorial deformed metric uses

```text
Delta_t = r^2 Delta_u
Delta_r = Delta_t [1 + log(1 + 6 nu/r^2 + 2 nu(26-3 nu)/r^3)]
Lambda_t = (r^2+a^2)^2 - a^2 Delta_t
xi = sqrt(Delta_t Delta_r)/(r^2+a^2)
p_r = p_r_star/xi
```

The code evaluates the full aligned expressions (6.1)-(6.5) of
Barausse–Buonanno (2010), including frame dragging, both spin couplings and the
mapped-spin-squared contribution. Radial derivatives of the metric are
analytic and include the derivative of `Delta5`'s `log(u)` term.
`metric_potentials` returns the lapse squared, radial metric, and `xi`; in the
spinning case the first of these is not simply `Delta_t/r^2`.

The inherited constructor and conservative derivative wrapper provide parameter
bookkeeping and Hamilton's equations. The prototype's metric and Hamiltonian
formulas are overridden. The inherited real-Hamiltonian interface includes
rest mass. `reduced_binding_hamiltonian` returns `(H_real-M)/mu` using a
rationalized expression that is finite at `nu=0`. Its additive constant does
not change the equations of motion.

## Inconsistent printed formulas and adopted interpretations

**Log-potential prefactor.** Bohé (2.2) prints `log(...)`, without the factor
`nu` multiplying it. Both the referenced Taracchini (2013) Eq. (2) and
Steinhoff (A1) explicitly give `nu log(...)`. The latter form also reproduces
the stated PN expansion and exact Kerr limit. The code uses this independently
corroborated `nu log(...)` form. Taking Bohé (2.2) literally with the cited
coefficients would violate those limits.

**Tortoise quartic.** Taracchini (2012) Eq. (14) transforms the radial quartic
with a factor `D^2 (r^2+a^2)^4/Delta_t^4`. Its prose then says to discard terms
above 3PN, but printed Eq. (15) retains `(r^2+a^2)^4` while dropping the
denominator. At `a=0` that introduces an extra factor `r^8` and fails the
nonspinning weak-field limit. The code instead performs the stated strict 3PN
truncation:

```text
Q4 = 2 nu (4-3 nu) p_r_star^4/r^2
```

This is also explicitly given by Damour–Nagar Eq. (7) and Steinhoff (A5) in the
nonspinning case. Spin-dependent terms introduced by expanding the tortoise
factor beyond its leading value lie beyond this quartic's 3PN order and are
discarded here. The primary evidence establishes this PN-consistent choice;
it does not establish whether a particular v4 implementation retains any
additional strong-field resummation of the quartic. An unambiguous
author-provided mathematical prescription would be needed to certify that
implementation detail without implementation-source or oracle inference.

## Validation

`tests/test_clean_room_seobnrv4_hamiltonian.py` covers:

- Formal expansion of the log-resummed potential through 4PN for three
  mass/spin configurations, including symbolic `log(u)` evaluated at two
  independent values. The target coefficients include the separately published
  4PN linear- and quadratic-in-`nu` terms.
- Exact equatorial Kerr metric, circular geodesic energy, orbital frequency,
  and vanishing radial acceleration at `nu=0`, with `a=-0.9,0,0.9`.
- The nonspinning tortoise Hamiltonian, including nonzero radial momentum and
  the independent 3PN quartic expression.
- Leading weak-field spin-orbit coupling
  `L (2 S_Kerr + 3 sigma_star/2)/r^3`.
- Analytic radial potential derivatives against automatic differentiation,
  and finite-difference gradient checks in all six mass/spin/state inputs.
- Scalar/batch agreement and inference-mode execution with inference tensors.
- Shared scalar coordinates with batched binary parameters, including outer
  parameter derivatives. Independent review found and corrected an inherited
  force-wrapper defect: coordinates must be expanded before creating the
  differentiation nodes, so forces are not summed across binaries.

The Hamiltonian, conservative, initial-condition and waveform tests passed:

```text
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_clean_room_seobnrv4_hamiltonian.py \
  tests/test_clean_room_eob_conservative.py \
  tests/test_eob_hamiltonian.py tests/test_eob_dynamics.py \
  tests/test_clean_room_eob_waveform.py
40 passed
```

After exposing the conservative-model option, the waveform suite separately
passed all 11 cases, including total-mass similarity for both selections.

`git diff --check` also passed. No binary-oracle comparison was performed by
this implementer. Validation establishes these equations, physical limits,
and differentiation behavior; it does not certify a complete waveform.

## Remaining limits

The class remains an explicitly selected experimental option. This work does not
replace radiation reaction, multipolar waveform amplitudes, nonquasicircular
calibration, merger attachment, or ringdown with their full v4 prescriptions.
Published calibration decimals may differ from unavailable extra precision.
No generic-spin dynamics is claimed.

Float64 spin derivatives of the log potential suffer subtractive cancellation
at extremely large radii, where the physical spin coupling is tiny: the
leading-spin test uses `r=10^4`, where truncation error is resolved, rather than
asserting accuracy at `r=10^5`. Potential values and ordinary inspiral radii
are not affected by that particular derivative test limitation. The metric
is intended outside its effective horizon; this reconstruction does not add
floors to continue square roots through invalid metric regions.
