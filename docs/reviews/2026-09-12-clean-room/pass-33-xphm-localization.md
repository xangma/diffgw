# Pass 33: IMRPhenomXPHM residual localization

Date: 2026-09-13  
Scope: clean-room localization of the strict-catalog and holdout XPHM residual.

## Boundary and method

This pass used only TorchWave source, the published waveform papers, and the
public `lalsimulation` waveform API as a black box. No LAL, PyCBC, or
TEOBResumS implementation source was inspected or copied. No fitted runtime
constant was introduced.

The probe used one fixed precessing case:

```text
m1,m2 = 35,25 Msun
S1 = (0.2,0.1,0.2), S2 = (-0.1,0.15,-0.1)
distance = 500 Mpc, inclination = 0.7, phi_ref = 0.3
f_ref = f_min = 20 Hz, delta_f = 0.25 Hz, f_max = 512 Hz
```

TorchWave and public LAL XPHM frequency-domain polarizations were compared
with complex relative L2 error and complex correlation. Public LAL mode-array
selection was used as an operational component probe. For a non-22 selection,
the selected `(2,2),(l,m)` waveform was differenced from the selected 22-only
waveform. This isolates the API's selected contribution; it is not treated as
an independently reconstructed inertial-frame mode.

## Results

The full waveform comparison was:

| comparison | plus relative L2 | cross relative L2 | plus correlation | cross correlation |
|---|---:|---:|---:|---:|
| all supported XPHM modes | 0.152528 | 0.157206 | 0.988596 | 0.987912 |

The dominant-carrier isolation gives:

| comparison | plus relative L2 | cross relative L2 |
|---|---:|---:|
| TorchWave XPHM 22-only vs public LAL XPHM 22-only | 0.048649 | 0.061171 |
| public LAL XPHM 22-only vs public LAL XP | 0.042301 | 0.053757 |

Thus, a material portion of the total residual is present before adding the
higher-mode projection assembly. The public LAL XPHM-versus-XP comparison is
an independent black-box control: it shows that the XPHM 22-only carrier is
not numerically identical to the XP carrier for this precessing case. The
TorchWave 22-only residual is of the same order, so the evidence points first
to the underlying carrier, remnant, or precession prescription rather than to
a uniquely identifiable Euler-angle defect.

The selected higher-mode probes were:

| selected contribution | plus relative L2 | cross relative L2 | plus correlation | cross correlation |
|---|---:|---:|---:|---:|
| `(2,1)` | 1.76005 | 1.81945 | 0.369939 | 0.343301 |
| `(3,3)` | 3.13963 | 3.16267 | 0.296247 | 0.293277 |
| `(3,2)` | 1.11768 | 1.09203 | 0.395457 | 0.432215 |
| `(4,4)` | 1.87794 | 1.89191 | 0.505145 | 0.502975 |

These large component residuals do not by themselves establish a rotation or
projection bug. Each selected contribution contains the clean-room coprecessing
carrier, the public model's mode selection semantics, and the subsequent frame
assembly. The per-mode result therefore cannot separate a carrier error from
an Euler-angle or spin-weighted projection error without an independently
validated coprecessing mode reference. Existing clean-room Wigner-d group-law,
aligned-continuity, C4 phase, and per-mode SPA-frequency tests pass and give no
source-backed rotation defect to patch.

## Decision

No runtime change is justified in this pass. The 22-only control establishes a
non-negligible carrier-level discrepancy, while the higher-mode decomposition
does not uniquely identify a projection defect. Changing an Euler-angle sign,
phase, or mode assembly factor would therefore be speculative and could mask
the larger carrier mismatch. The quantitative blocker is the absence of an
independent coprecessing XHM carrier oracle that is separable from the public
LAL projected output under the clean-room constraints.

Validation after report-only changes:

```text
python -m py_compile src/torchwave/approximants/imrphenom_xphm.py
git diff --check
```

