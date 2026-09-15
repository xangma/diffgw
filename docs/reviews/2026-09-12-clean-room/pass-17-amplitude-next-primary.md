# Pass 17: source-complete XHM amplitude audit

Date: 2026-09-13. This is a bounded primary-source audit of the native XHM
amplitude and ringdown construction. It does not use implementation ports,
generated C, notebooks, binaries, residuals, or verification output, and it
makes no runtime change.

## Evidence and provenance

The primary source is García-Quirós et al.,
[arXiv:2001.10914v2](https://arxiv.org/abs/2001.10914v2), revised 3 November
2020. The arXiv source archive used for the equations is
[the v2 source](https://arxiv.org/src/2001.10914v2), SHA-256
`cc05e7dbff7bf0c42a10b7da1f4840eefc55b252168d6bfb085155a7bdc5318e`; its
`hmc2.tex` is SHA-256
`5c7132720db24d9caaa804f53d6961b075996cacee00b3ce0a51865a6dbf107e`.

The retained ancillary evidence is the official author archive under
`anc/ParspaceFits`. The archived README is SHA-256
`368184fb645eb93dc04963902c61bd37fd6be8be69be1842abed7994c20684f4` and says
that amplitude coefficient data are consumed by the author amplitude notebook;
that notebook and generated C are excluded here. The archived mathematical
phase assets are not amplitude assets, but their provenance and hashes are
recorded for completeness:

| asset | official arXiv URL | SHA-256 |
|---|---|---|
| `3Dfit_p6_lm_21.m` | [21 asset](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/21/3Dfit_p6_lm_21.m) | `5b3ab0e7f04092f99f8983b02f7cb985cf4bfd13b3e4022b16cf020d57987716` |
| `3Dfit_p6_lm_33.m` | [33 asset](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/33/3Dfit_p6_lm_33.m) | `7cb4273b6b46f4be4f82b5a48e03705fa0f1d832effd71c79ade845c2d91afe8` |
| `3Dfit_p6_lm_44.m` | [44 asset](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/44/3Dfit_p6_lm_44.m) | `6dd15b7cf9fded8c3cdd329c7535d688f3d25633a9508c00ebf7ad2b29d5c8f` |

## Published amplitude and ringdown behavior

The paper defines the amplitude inspiral cut in Eq. (4.4), source lines
572--581:

```text
f_Ins^(lm) = (m/2) [ f_MECO^22 epsilon_Ins^(lm)
                     + |f_ISCO^22 - f_MECO^22| delta_Ins^(lm) ].
```

Table II, source lines 592--614, gives the mode-specific functions. For the
32 mode the literal v2 table entry is

```text
delta_Ins^32 = (3/4 - 0.235 |chi_eff|) (f_ring^32 / f_ring^32),
epsilon_Ins^32 = f_ring^32 / f_ring^22.
```

The first ratio is therefore exactly one as printed. The native code currently
uses `del_amp["32"] = (0.75 - 0.235*abs(chi_eff)) *
(f_ring["32"]/f_ring["22"])` at
`src/torchwave/approximants/imrphenom_xhm.py:718-723`, while its
`eps_amp["32"]` at `:711-716` agrees with the table. Thus the native 32
amplitude inspiral cut differs from the literal published equation whenever
`f_ring^32 != f_ring^22`. The discrepancy is confined to the amplitude cut;
the phase cut remains the separate `f_Ins^(lm)=(m/2)f_MECO^22` construction.

The remaining inspected amplitude construction is source-consistent:

* The fifth-order inverse polynomial and its two equally spaced intermediate
  nodes are Eq. (5.3), source lines 826--839. The native nodes at
  `:1060-1065` and reciprocal `p` boundary construction at `:1073-1114`
  implement those definitions.
* The ringdown interval is `Mf in [f_RD^lm, 0.3]`, source lines 1049--1053.
  The cut factors in Table IV, source lines 801--823, match native
  `f_rd_amp` at `:730-735`.
* The ringdown ansatz, Eq. (6.4), source lines 1087--1091, is

  ```text
  A_RD^lm/A_0^22 = f^(-1/12) |a_lambda| f_damp^lm sigma /
      ((f-f_ring^lm)^2 + (f_damp^lm sigma)^2)
      exp(-(f-f_ring^lm) lambda/(f_damp^lm sigma)).
  ```

  Native `eval_h_rd` at `:913-921` has the same factors and mode-specific QNM
  frequencies. Table V, source lines 1098--1117, fixes sigma to 1.30, 1.33,
  and 1.33 for modes 33, 32, and 44; native `sigma_amp` at `:747-753`
  matches those constants while retaining a fit only for mode 21, as the text
  requires.
* For mode 32 the paper requires fitting in the spheroidal basis and
  transforming the full waveform back, source lines 1055--1075. The native
  path constructs the mixed complex ringdown before taking its spherical
  magnitude at `:1015-1038`; no separate source-complete amplitude change was
  found in this audit.

## Typographical-error audit and implementation decision

The repeated denominator in the printed 32 delta factor was audited against
the complete allowed v2 TeX source and the retained official mathematical
assets. The only occurrence of this factor is the Table II entry quoted above
(source lines 592--614). No other equation, table caption, nearby discussion,
or allowed `.m` asset states `f_ring^32/f_ring^22` for `delta_Ins^32`, or gives a
different denominator. Equation (4.4) (source lines 572--581) treats delta and
epsilon as separate mode-specific functions, so literal transcription makes
the delta ratio one while the independently printed epsilon ratio remains
`f_ring^32/f_ring^22`. The large-q local-maximum replacement (source lines
874--881) is a separate branch and does not resolve this denominator.

The repeated `32` could be a typographical error, but that conclusion would
be an inference from the appearance of the table rather than corroboration by
an allowed primary source. Therefore the ratio-removal patch is **not
authorized by this audit**: do not change native `del_amp["32"]` in pass 17.
The native expression differs from the literal table, and the discrepancy
should remain documented pending a primary erratum or an official author
amplitude asset that resolves the denominator. No other source-complete
amplitude discrepancy was established here.

## Conditional test plan if the denominator is later resolved

If an allowed primary source later resolves the denominator to `32`, add a
direct arithmetic test around the amplitude-cut calculation (or a small pure
helper if the calculation is factored) with a parameter row for which
`f_ring32/f_ring22 != 1`. Assert that the mode-32 amplitude cut is

```text
(2/2) [ f_MECO22 (f_ring32/f_ring22)
        + |f_ISCO22-f_MECO22| (0.75-0.235|chi_eff|) ],
```

and that the current extra ratio on the second term is absent. Check the
published 21, 33, and 44 cuts and the phase cuts remain unchanged. Run the
same assertions batched in float32 and float64, and differentiate the cut
with respect to a spin input to ensure the arithmetic remains differentiable.
The test should use only native tensors and published equations; it should
not compare against an implementation, binary, residual, or calibrated
waveform.

No runtime or test file was edited in this audit.
