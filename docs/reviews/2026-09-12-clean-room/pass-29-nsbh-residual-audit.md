# Pass 29: NSBH residual and phase-gauge audit

This pass used the primary PhenomNSBH and NRTidalv2 papers, TorchWave's
equations, and public `lalsimulation` waveform calls as a black-box oracle. It
did not inspect LALSuite or PyCBC implementation source, and no residual was
used to fit a runtime coefficient.

Primary references are Thompson et al., [arXiv:2002.08383](https://arxiv.org/abs/2002.08383),
and Dietrich et al., [arXiv:1905.06011](https://arxiv.org/abs/1905.06011).

## Branch and remnant measurements

The current equations classify the strict catalog point
`(m1,m2,chi1,Lambda2)=(5,1.4,0.2,400)` as non-disruptive: `f_tide =
2736.9320 Hz`, `f_RD = 2562.4719 Hz`, and the torus fit is zero. The holdout
point `(7,1.4,0.4,2600)` is disruptive with a torus ratio `0.1176731`:
`f_tide = 1440.8683 Hz < f_RD = 2019.7245 Hz`. The Zappa remnant values used
by the amplitude are `(M_f, chi_f)=(6.2183543,0.5971823)` and
`(8.0449878,0.6252066)`, respectively. These values follow the published
classification and remnant workflow; no alternate branch is source-defined
by the residuals.

## Strict black-box residuals

At `df=0.25 Hz`, `[30,1024) Hz`, the catalog complex relative L2 is
`2.9516980e-5`; its amplitude RMS/max relative errors are `8.1031e-7` and
`2.9796e-6`. The phase residual is an affine carrier to numerical precision:
the best-fit slope is `-2.9438889e-7 rad/Hz`, with detrended RMS
`3.07e-13 rad`. Thus this point does not identify an amplitude, branch, or
intrinsic phase defect.

For the holdout, the complex relative L2 is `1.7579727e-3`; amplitude
RMS/max errors are `4.4833e-8` and `1.2043e-7`. The raw phase residual has
best-fit slope `-1.9586591e-5 rad/Hz`; after removing the affine carrier its
RMS/max are `1.0092e-5` and `1.0832e-4 rad`. The residual therefore remains
overwhelmingly a time/phase gauge difference, with a small nonlinear remainder.

## Tidal-phase isolation

For the holdout, changing `Lambda2` from zero to 2600 in the standalone
black-box `IMRPhenomD_NRTidalv2` call gives the same phase change as TorchWave's
`-117 Lambda_tilde x^(5/2) P(x)/(256 eta)` expression at 30, 40, 100, 300,
500, 800, and 1000 Hz (differences below `1e-13 rad`). This is the
`39 kappa_eff/(16 nu)` form of Dietrich et al. Eq. (17), with
`kappa_eff=3 Lambda_tilde/16`. The same test against `IMRPhenomD_NRTidal`
selects the distinct v1 curve, so the v1/v2 choice is not ambiguous here.

The full `IMRPhenomNSBH` oracle has an additional Lambda-dependent carrier
shift relative to that standalone tidal phase. TorchWave's default inherited
gauge reproduces the shape but with a different affine scale, leaving the
holdout residual above. Thompson et al. specify the PhenomD plus NRTidal
intrinsic composition, but do not specify this global affine phase/time anchor;
the current code explicitly labels that path as a compatibility gauge. Changing
it from the oracle residual would be an underdetermined fit, not a clean-room
source correction.

## Outcome

No runtime change is justified in this pass. The remaining blocker is the
underdetermined default NSBH affine phase gauge (and its tiny nonlinear
implementation remainder), while low-Lambda notation and calibration-digit
provenance remain separately unresolved. Existing source-backed branch,
frequency, remnant, amplitude-window, and NRTidalv2 equations are retained.
