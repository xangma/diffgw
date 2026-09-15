# XHM `(3,2)` phase target artifact audit

Date: 2026-09-14. Primary source inspected: author arXiv v2 archive,
`https://arxiv.org/e-print/2001.10914v2` (downloaded archive SHA-256
`cc05e7dbff7bf0c42a10b7da1f4840eefc55b252168d6bfb085155a7bdc5318e`),
including `anc/ParspaceFits/README` and `hmc2.tex`.

The archive separates intermediate phase fits from spheroidal ringdown fits.
`anc/ParspaceFits/PhaseFits/32/3Dfit_p1_lm_32.m` through `3Dfit_p6_lm_32.m`
are six intermediate-region phase collocation fits. Their SHA-256 hashes
(p1 through p6) are:

```
6b3cdca278a4f9218c12f348d780295655e3ab4a0d06d8f53386beac79be595e
eb28bab3fdcf6e9c806746a454e82e22421eb2b26594dc071f18c42daea2b9ab
e908d8812f7f189c4e21d1612fdf502eb824243cf5ead9d629041f4bf6cb4950
3004e177139c565f4783f2a57c053264f31aff73b474ad924171b141b0fe80c2
2313f19639896287967cb2c9a1365fc4bb6757f65efff91c9d55090d247a372e
ebf98846c5d9e6a6888766e5dc7d33a7e7b5e2b45aed718acf5438ab317eb516
```

The spheroidal ringdown directory contains three derivative-target files:
`dphiS_p1.m` (SHA-256
`3c6ad2109a6d85202f794579c88b9b52ebe9b39340e1aa9f02ef4fa1f01b89d7`),
`dphiS_fring32.m` (`1a1b3519df07ffef8a0a065184a3ce66417895569ba540852877b8f7c4ca3ddf`),
and `dphiS_p4.m` (`c417a56b22950dae88c782f35d4470b22af92073cffe92dc273e8e71c155d082`).
The other two files, `deltaPhiS.m` (`e92a6657458934bea6bb8466b97c7cd90a45d57d98eff9b546e7d64804fac772`)
and `deltadPhiS.m` (`07107d0436b316792e0b2cd343b8cce3efa2a96276a26304f6ecbdfb7c975455`),
are alignment quantities. The README identifies the notebook as containing
the phase fits, while direct inspection of these file contents shows the
three `dphiS` expressions are derivative fits and the delta files are phase
and derivative alignment expressions. Thus the archive does not establish
four independent spheroidal derivative targets required by Eq. (48).

The prior four-file provider was removed because it mislabeled intermediate
`3Dfit_p1..p4` surfaces as spheroidal ringdown targets. No numerical provider
or production change is added. The archive is paper-era, does not identify
release `122022`, and does not state redistribution terms clearly enough to
authorize bundling.
