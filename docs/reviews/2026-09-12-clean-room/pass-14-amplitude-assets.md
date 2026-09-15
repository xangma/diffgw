# Pass 14: XHM amplitude calibration assets

## Scope and sources

This pass audits the author source for the intermediate amplitude target and the
native XHM mapping. The requested local files
`/tmp/torchwave-clean-higher-modes/2001.10914v2.{pdf,txt}` were not present in
the workspace. I used the primary arXiv v2 source instead:

* [arXiv:2001.10914v2 source](https://export.arxiv.org/e-print/2001.10914v2),
  downloaded as `/tmp/xhm-author-source/xhm.tar`, SHA256
  `cc05e7dbff7bf0c42a10b7da1f4840eefc55b252168d6bfb085155a7bdc5318e`.
* `hmc2.tex`, SHA256
  `5c7132720db24d9caaa804f53d6961b075996cacee00b3ce0a51865a6dbf107e`.
  The downloaded paper PDF has SHA256
  `00f3bdbe28a6ccde5cde3af82ac8fc95153ec6f03a4b071ce0ed57326e4e09ee`.
* `anc/ParspaceFits/README`, SHA256
  `368184fb645eb93dc04963902c61bd37fd6be8be69be1842abed7994c20684f4`.
  It identifies `AmpFits` as the coefficient data used by the author fit
  notebook. The notebook itself was not read.
* Native files audited: `src/torchwave/approximants/imrphenom_xhm.py`, SHA256
  `ed2f42f4a52fbafe1f42c90f94a3a24585e838cb95350d0f521eb5be1e60fd04`, and
  `src/torchwave/approximants/xhm_fits.py`, SHA256
  `6704f0c59a1e1527f5c31063d1bac8ab84f7349207acc3f1c24532c370167548`.

The eight inspected author assets are the `CoefficientStatsInt1.m` and
`CoefficientStatsInt2.m` files under `AmpFits/{21,32,33,44}`. Their SHA256
hashes are, respectively:

```
21/Int1 e0c25f0119cffdf50552a4e9d78a92470f0e4c28e57f83ed94aad7d5d1427fdf
21/Int2 80be60a0c5159263d3190e2b93e08a208d199cd66095668814607a1bf4cad723
32/Int1 f5882de942f2f23dcffc550b7974152cabe110d5af86c32eb186f06a8cdbe561
32/Int2 d5ac1aabf9d76117942afa47907dd6a8dab1d084d2c65594ee92796c98cbba2d
33/Int1 1b79d8a6eafff0816365dd83dfbed33a2879d2f92c05d096c518396e06a6b7e2
33/Int2 b43e4b5e2e05310440938b42e89a68886d0aa5e6110e3988df04c0ea77f120b6
44/Int1 36dc12cce856046df1336bdac11d8da5ea51746a445e75f0e35652f9dd686188
44/Int2 a31ffbf07d27f7e8799541afeb8f45c274c6c28d00b87ee53b1c77e781f8637b
```

Exact copies of those eight files, the ancillary README, and short relevant
`hmc2.tex` excerpts are retained in the ignored local review cache at
`docs/reviews/2026-09-12-clean-room/source-assets/pass-14-amplitude/`. The
fixture records the source URL, hashes, selected `3DFits` entry, and generation
script. The arXiv source is used under its nonexclusive distribution license;
the cache does not imply relicensing.

## Author conventions

The paper defines `q=m1/m2 >= 1`, `M=m1+m2`,
`eta=m1*m2/M^2`, and dimensionless aligned spins
`chi_i=(S_i dot L)/(m_i^2 |L|)`. With normalized component masses,
`m1=(1+sqrt(1-4 eta))/2` and `m2=(1-sqrt(1-4 eta))/2`, so the native

```
S = (m1^2*chi1 + m2^2*chi2)/(1 - 2*eta)
```

is exactly the paper's `S-hat` definition. The paper uses `chi_PN` for
inspiral spin terms and `S-hat` for merger-ringdown terms. It also states that
odd-m amplitude leaves are sign-flipped before fitting to remove a crease,
then the absolute value of the final fit is taken. The invariant
nonspinning/equal-spin part of odd-m fits carries a `sqrt(1-4 eta)` factor and
vanishes at equal mass. Terms proportional to `chi1-chi2` can remain at
`eta=1/4` with unequal spins, so odd modes need not vanish for equal masses
with unequal spins. This preprocessing does not establish the amplitude
target's numeric units.

For inspiral, the source defines

```
A22^0 = pi*sqrt(2 eta/3)*(pi f)^(-7/6)
H_lm = |A_lm^SPA|/A22^0
```

and fits three collocation values of `H_lm`. For the default intermediate
region (paper Eq. `AmpInter`, source lines 832--835), it defines

```
A_lm^Inter/A0 = 1/(delta0 + delta1*f + ... + delta5*f^5)
A0 = pi*sqrt(2 eta/3),       # frequency independent
```

with collocation values at one-third and two-thirds of the interval from
`f_Ins` to `f_RD`. Therefore, if `v=A_lm^Inter/A0`, the corresponding
`H_lm` value is `v*(pi*f)^(7/6)`. The native polynomial is intentionally in
the reciprocal variable `p=(pi*f)^(7/6)/H_lm`; consequently `p=1/v` at an
intermediate collocation point and `h=(pi*f)^(7/6)/p` after reconstruction.

## Mapping evidence and decision

Each inspected `CoefficientStatsInt{1,2}.m` record contains `NSFit`,
`1DSpinFit`, `3DFits`, and `Best3DFit` metadata. The archived `3DFits` records
whose full expressions match the native coefficient transcription are used
for the independent samples (entry map is recorded in the test). This is
formula/coefficient correspondence only. The README says that `AmpFits`
contains coefficient data, but neither it nor these `.m` records names the
fitted observable or specifies all table preprocessing. The paper supplies
the target normalization and describes the odd-mode sign/absolute-value
handling; formula agreement alone does not prove the inherited table's exact
data variable or preprocessing.

The reciprocal mapping at `imrphenom_xhm.py:925-941` and `:1090` agrees with
the `A^Inter/A0` target for positive fitted values. The independent fixture
also exposes negative odd-mode fit values; the paper requires taking the
absolute value after such sign-folded fits. Consumer handling of that
absolute-value requirement is left to the separate runtime pass. No oracle or
translated implementation was used. Independent FP64 evaluations of all eight selected
author expressions over equal-mass unequal-spin and mass-ratio/spin samples
are in `tests/data/xhm_amplitude_author_samples.json`. The generator does not
import native XHM code. Parent integration changed the ordinary regression to
compare native raw fits with these independent values without requiring the
ignored source cache. This asset review makes no runtime change; the separate
[odd-mode correction](pass-14-odd-amplitude.md) applies the absolute value at
the production consumer.
