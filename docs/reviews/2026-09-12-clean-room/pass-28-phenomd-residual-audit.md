# Pass 28: clean-room IMRPhenomD residual and convention audit

Date: 2026-09-13

This pass audits the remaining strict IMRPhenomD residuals without reading LAL or
PyCBC implementation source. The only external waveform values used here were
black-box results from the public `lalsimulation.SimInspiralChooseFDWaveform`
API. The primary sources are Khan et al., [arXiv:1508.07253v2](https://arxiv.org/abs/1508.07253v2),
Husa et al., [arXiv:1508.07250v2](https://arxiv.org/abs/1508.07250v2), and the
coefficient/QNM provenance audit in
[pass 18](pass-18-phenomd-assets-primary.md).

## Residual decomposition

For each frozen case, TorchWave and the public LAL PhenomD output were evaluated
on the stored `df=0.25 Hz` grid, retaining positive nonzero bins. Let
`r(f)=h_TW(f)/h_LAL(f)` and unwrap `arg(r)`. The amplitude column is
`|| |h_TW|-|h_LAL| ||_2 / ||h_LAL||_2`. The phase columns come from a weighted
least-squares fit

```
arg(r(f)) = phi0 + 2*pi*f*dt + nonlinear(f),
```

with weights `|h_LAL|^2`; `phi0` is the constant phase residual and `dt` is the
affine time-shift residual. The nonlinear columns are the RMS and maximum of the
remaining phase. These are diagnostics, not fitted runtime corrections. The
complex column is the direct relative complex L2 error.

| case | complex L2 | amplitude L2 | phi0 (rad) | dt (s) | nonlinear RMS (rad) | nonlinear max (rad) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| catalog 30+20, `chi=(.3,-.2)` | 4.9476516e-7 | 7.5388189e-8 | -1.7971705e-7 | 1.2631002e-9 | 6.5075842e-8 | 9.2252156e-7 |
| holdout0 57+13, `chi=(.83,-.41)` | 1.2321902e-5 | 1.5598414e-6 | -6.5819270e-6 | 3.7337945e-8 | 1.4649509e-6 | 2.8757728e-5 |
| holdout1 23+21, `chi=(-.6,.2)` | 8.9629199e-7 | 1.4005178e-7 | -5.4609903e-7 | 2.4904695e-9 | 4.3052766e-8 | 1.1629558e-6 |
| holdout2 80+8, `chi=(-.7,.1)` | 9.3693732e-7 | 1.7769823e-7 | -7.3839406e-7 | 5.7822819e-9 | 6.4576455e-8 | 5.8489648e-6 |
| holdout3 42+42, `chi=(.92,.92)` | 1.5435514e-5 | 1.8602506e-6 | -6.0435441e-6 | 3.7071330e-8 | 2.5811715e-6 | 3.0753880e-5 |

The catalog residual is almost entirely below one micro-radian in phase and
`8e-8` in amplitude. The two larger high-spin/equal-spin residuals retain a
frequency-dependent phase component after removing the best affine gauge. Thus a
constant phase or time shift cannot explain the holdout failures. The cross
polarization gives the same decomposition up to floating-point noise because the
current nonprecessing polarization factors share the same phase.

## Public API convention probes

The black-box LAL API confirms the phase conventions that can be tested without
implementation access:

| probe | observed public API behavior |
| --- | --- |
| `phiRef` | Changing `phiRef` by `delta` multiplies the complex waveform by a constant phase `+2*delta`. TorchWave has the same transformation for `phic`. |
| `f_ref` | LAL changes only a constant phase: relative phases for 40 versus 20 Hz and 80 versus 20 Hz were `-0.5508165972` and `-0.0956171044` rad, with unit amplitude and zero slope. TorchWave likewise produces constant ratios (`-0.5508167350` and `-0.0956175177` rad). The small offsets are model residuals, not evidence for a new reference-frequency law. |
| `tc` | `SimInspiralChooseFDWaveform` has no coalescence-time argument. Its nearby positional argument is `longAscNodes`; varying it is not a valid `tc` probe. TorchWave's explicit `tc` transformation is `h -> h exp(-i*2*pi*f*tc)`, covered by local phase-convention tests. |
| high-frequency support | For 80+8, the public LAL array became exactly zero after about 461.0 Hz at this grid spacing, while TorchWave's current boundary retained the 461.25 Hz bin. For 30+20 both remained nonzero through the requested 512 Hz range. This is a boundary/convention diagnostic only. |

The primary papers do not uniquely define a hard `Mf=0.2` cutoff. Husa's
construction describes the full piecewise model and discusses a separate
high-frequency taper used for time-domain conversion; it does not publish the
boundary rule observed in a particular API build. Therefore the one-bin cutoff
difference cannot justify a runtime change under the clean-room constraint.

## Source-complete correction decision

No runtime patch is made in this pass. The allowed primary archives contain no
machine-readable full-precision Table V coefficient asset. Their printed values
are rounded, and pass 18 records the unresolved accounting ambiguity between the
17 analytically constrained quantities and the 19 printed rows. Husa identifies
Berti data interpolation for the ringdown frequency but does not publish the
knots, interpolation order, endpoint policy, or a machine-readable table in the
allowed archive. The papers also permit arbitrary global time and phase shifts
without specifying the absolute anchor used by a public API. Finally, the hard
high-frequency cutoff is not source-defined.

Consequently, replacing coefficients, changing the QNM interpolation, altering
`f_ref`/`tc` anchoring, or moving the cutoff would be an underdetermined
calibration or convention choice. The residual decomposition provides evidence
for a nonlinear model difference in the high-spin holdouts, but no uniquely
published equation or independent author asset identifies its correction. The
existing runtime and test files are therefore unchanged; this report is the
only artifact added by pass 28.
