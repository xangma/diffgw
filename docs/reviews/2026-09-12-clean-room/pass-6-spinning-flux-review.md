# Independent review of the published spinning flux

No actionable implementation defect was found in `seobnrv4_flux.py` or in
the direct-square flux changes in `factorized_modes.py`. This conclusion is
limited to the declared published-equation reconstruction. It does not certify
SEOBNRv4 implementation parity or remove the explicit ell=8 truncation choice.
No implementation or test file was changed during this review.

## Scope and primary sources

The review independently read the complete new flux module, the shared
Newtonian prefactors, spin regularization and flux assembly, the Hamiltonian
adapter's relevant normalization and broadcasting, and the focused tests.
The previously reviewed DIN residual table and Bohé 22 amplitude were reused;
the new nonspinning extensions and spin coefficients were checked directly.
No external waveform source, source port, binary oracle, oracle output, or
historical calibration material was accessed.

The following primary PDFs were inspected; their local hashes were recomputed
and match the implementation report's ledger:

| Primary publication | Equations checked | PDF SHA-256 |
|---|---|---|
| [Pan et al., 1006.0431v2](https://arxiv.org/pdf/1006.0431v2) | (3), (28), (29), (30), (33), (38), (40), (41), (D1), (F26) | `4b51940aae4299c129b08d64f7f5ae92a5f9de00e96f2653abe719e76d596c19` |
| [Taracchini et al., 1202.0790](https://arxiv.org/pdf/1202.0790) | (12), (13), (A1)–(A6), (A8b)–(A12) | `d6d126be90bc789835ec363451f4ec51728a8d6ddc09eff078620fdc46860745` |
| [Taracchini et al., 1311.2544](https://arxiv.org/pdf/1311.2544) | (4) and its amplitude inheritance prescription | `f57b099e701c1206802d9887f6843ce75fc8753ac9c3ae5c2137933160f85967` |
| [Buonanno, Faye and Hinderer, 1209.6349v2](https://arxiv.org/pdf/1209.6349v2) | (4.26), (4.27a), phase-convention text | `1252291e97215e07fb92a028d8faffb6bcd95122b895312c6692a993fd318af0` |
| [Bohé et al., 1611.03703v3](https://arxiv.org/pdf/1611.03703v3) | (2.1), (2.9), exclusion of NQC from radiation reaction | `db9dc054368f131bdc152b319e54441c665547668221891723b309b005ac23ba` |

## Findings checked

- All nine additions to the nonspinning residual table agree with Taracchini
  Appendix A. Keeping the DIN/Pan sign of the 55 v² coefficient is justified:
  the opposite sign printed in Taracchini (A11a) fails Pan (D1a)'s negative
  test-particle coefficient.
- The ordinary Kerr rho coefficients and odd-m residual-f coefficients match
  Pan (28), (29), and (D1), including the cubic-spin 41 term. Finite-mass
  replacements are not added a second time to the replaced Kerr terms.
  The spin variable is the stated total spin per M².
- An independent symbolic reduction starts with BFH's unnormalized
  `hhat21 = -43 delta S/21 + (-79+139 nu) Sigma/42` and
  `S=(1-2nu)chiS+delta chiA`, `Sigma=-delta chiS-chiA`.
  Accounting for the published phase-origin conversion and dividing by
  Pan's Newtonian 21 mode gives `b3=3 hhat21/delta`.
  Subtracting `j2 f1+j3` from Pan (40) produces exactly the implemented
  `[(427+79nu)chiS+(427+131nu)chiA/delta]/84`; the symbolic difference
  vanishes using `delta²=1-4nu`.
- For 32, Pan (F26) gives `rho_v2_spin=-rho_v1_spin²`. Cubing rho therefore
  cancels the otherwise spurious spin-square contribution at v². The
  polynomial treatment of `c/delta` also preserves the odd-mode equal-mass
  limit without an evaluated mass-difference denominator.
- Taracchini (A2) fixes the zero-radial-frequency derivative at the same
  radius and angular momentum, as the adapter does. Only the Newtonian
  prefactor uses the resulting non-Keplerian velocity. The odd source uses
  `p_phi v`, and the tail uses total rest-mass-inclusive real energy.
- The positive-m flux factor `1/(8 pi)` accounts for the negative partners
  once. Reduced-mass torque is negative for positive emitted flux, and
  the radial-force ratio follows the momentum-directed force prescription.
- Replacing `abs(h)²` by `Re(h)²+Im(h)²`, and passing prefactors before their
  absolute value, preserves flux values and restores the physical Hessian
  at exactly vanishing modes. This does not choose an additional mode phase.

The two ell=8 choices remain explicit publication-level reconstructions:
Pan's tabulated rows stop at v², while its general even-parity result provides
a v³ spin term. Neither the source wording nor these checks identifies a
particular external implementation's choice. The module correctly leaves
NQC and residual phases out of the radiation-reaction calculation.

## Independent validation

The focused suite passed **72 tests in 5.45 seconds**:

```text
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest \
  tests/test_clean_room_factorized_modes.py \
  tests/test_clean_room_seobnrv4_flux.py \
  tests/test_clean_room_eob_factorized_integration.py \
  tests/test_clean_room_seobnrv4_amplitude22.py -q
```

Separate review probes verified the BFH symbolic identity above, compared
the Hessian of two exactly vanishing complex modes with the analytic
quadratic-form Hessian (zero absolute difference), and evaluated a shared
scalar orbital state over a 2-by-3 Hamiltonian parameter batch. The latter
agreed exactly with six scalar evaluations. These probes used only primary
equations and Torch/SymPy algebra, not waveform reference implementations.
