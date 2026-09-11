The pasted analysis reproduces several useful smoke-test numbers, but it does not establish waveform parity. Some reported mismatches largely come from the match calculation; other cases hide substantial implementation defects or compare different models.

This review examined the current, dirty working tree at HEAD `1ed8d50db7fe6b2220f86e41082a2f79764f9330`. It changed no waveform implementation. LAL was used through its public binary API; no LAL waveform implementation source was inspected.

**What was reproduced.** All 28 selected existing LAL-comparison tests passed; 24 other tests in those files were deselected. An additional 18-case probe measured both polarizations, amplitude errors, excluded nonfinite bins, and two time/phase match calculations. Environment: Torch 2.9.1, NumPy 2.3.5, LAL 7.7.0, LALSimulation 6.2.0, PyCBC 2.10.0; CPU float64/complex128. The saved [probe](/Users/xangma/repos/torchwave/docs/reviews/2026-09-10-parity/probe.py), [results](/Users/xangma/repos/torchwave/docs/reviews/2026-09-10-parity/results.json), and [test log](/Users/xangma/repos/torchwave/docs/reviews/2026-09-10-parity/pytest.log) contain the evidence.

**The most serious missed defect is PhenomD cross polarization.** At face-on inclination, the implementation produces `h_cross = -i * conj(h_plus)`, whereas the LAL output satisfies `h_cross = -i * h_plus`. The real part has the wrong sign in [imrphenomd.py:862](/Users/xangma/repos/torchwave/src/torchwave/approximants/imrphenomd.py:862). This is a frequency-dependent phase error, not a removable constant polarization convention. The control probe verifies both identities directly. Across the ten D cases, cross matches are only about 0.115–0.183. NRTidal and NSBH inherit the affected carrier and also fail cross comparisons. The existing LAL tests compare plus only.

**The reported metric also needs correction.** The pasted commands use normalized, flat-noise PyCBC `match`, maximized over phase and discrete time shifts. This discards absolute amplitude information and does not measure samplewise equality. PyCBC documents only fourth-decimal accuracy for this routine and recommends `optimized_match` for more precise time optimization. Eight printed decimals do not establish machine precision. [PyCBC documentation](https://pycbc.org/pycbc/latest/html/pycbc.filter.html#pycbc.filter.matchedfilter.match)

The following are flat-noise diagnostics, with independent time/phase maximization for each polarization. The refined column uses `optimized_match`; it is still not a complete parity test.

| Comparison | Original-style plus match | Refined plus match | Original-style cross match |
|---|---:|---:|---:|
| TaylorF2, 1.4+1.4 solar masses, circular, 30–500 Hz | 0.999999999 | 0.999999999 | 0.999999999 |
| TaylorF2, same masses, lambdas 400/400, 30–1000 Hz | 0.999992289 | 0.999999298 | 0.999992289 |
| PhenomD, 30+10, spins 0.5/−0.3, 20–1024 Hz | 0.996349978 | 0.999999222 | 0.155066097 |
| PhenomD, 30+30, spins 0.8/0.8, 20–1024 Hz | 0.999334238 | 0.999943586 | 0.142758407 |
| D_NRTidal v1, 1.4+1.4, lambdas 400/400, 30–1000 Hz | 0.999241112 | 0.999999996 | 0.166098894 |
| D_NRTidal v2, pasted BNS parameters, 25–1200 Hz | 0.999616095 | 0.999996669 | 0.177931584 |
| NSBH, 6+1.4, BH spin 0.2, lambda2=450, 30–1024 Hz | 0.999154402 | 0.999948865 | 0.174034427 |
| Eccentric: original ineffective arguments, versus EccentricFD | 0.999999999 | 0.999999999 | 0.999999999 |
| Eccentric: corrected e=0.05 at 30 Hz, versus EccentricFD | 0.795105271 | 0.795470110 | 0.795105093 |
| Eccentric: corrected arguments, versus TaylorF2Ecc | 0.908547424 | 0.912237893 | 0.908547424 |

For example, the high-spin D waveform has a plus norm ratio of 0.989119 and magnitude relative L2 error of 0.014046 despite its high match. The nonspinning TaylorF2 example has amplitude relative L2 error about `1.23e-8`; “machine epsilon” remains unsupported. Every probed Torch waveform has a nonfinite DC bin, excluded by these positive-frequency match bands. The JSON raw complex residuals include unresolved reference-phase/time conventions and must not be interpreted as convention-corrected model errors.

**The eccentric “exact” result compared circular limits.** Torch expects `e0`, but the pasted call passes `eccentricity`, silently swallowed by `**kwargs`. LAL receives positional eccentricity zero; inserting 0.05 in its dictionary has no effect in this call. Controlled comparisons confirm both no-ops. Use `e0=0.05, f0_ecc=30.0` on Torch and positional eccentricity 0.05 on LAL. Also fix the zero-frequency default for `f0_ecc`. See [the eccentric interface](/Users/xangma/repos/torchwave/src/torchwave/approximants/taylorf2_eccentric.py:127).

The target must be explicit: the installed LAL canonical names include `TaylorF2Ecc` and `EccentricFD`. A partial low-eccentricity TaylorF2 correction does not implement the full enhanced post-circular model. These correspond to different constructions in [Moore et al.](https://arxiv.org/abs/1605.00304) and [Huerta et al.](https://arxiv.org/abs/1408.3406). The Moore citation in the file is wrong: the authors are Moore, Favata, Arun, and Mishra; the journal reference is Phys. Rev. D 93, 124061 (2016).

**Several catalog entries need substantial model work.**

| Entry | Verified state and implication |
|---|---|
| XAS | [The entire calculation delegates to D](/Users/xangma/repos/torchwave/src/torchwave/approximants/imrphenom_xas.py:60), with a different default taper. With the taper matched, both polarizations are exactly equal to D. It has no XAS-specific calibration. The published XAS is an overhaul of D, not this wrapper. [XAS paper](https://arxiv.org/abs/2001.11412) |
| HM / XHM | [XHM is a pass-only subclass](/Users/xangma/repos/torchwave/src/torchwave/approximants/imrphenom_hm.py:417); both default outputs are identical. Higher modes use leading PN ratios and simple suppression, not distinct XHM calibration. The published XHM includes calibrated higher modes and (3,2) mode mixing. [XHM paper](https://arxiv.org/abs/2001.10914) |
| HM phase | [Wrapped `torch.angle` is multiplied by `m/2`](/Users/xangma/repos/torchwave/src/torchwave/approximants/imrphenom_hm.py:348). A 2π branch jump becomes a sign flip for odd m. XPHM inherits this defect. Expose continuous analytic phase; sampled unwrapping alone is unreliable on coarse grids. |
| P / Pv2 / XP / XPHM | [P and XP use the same D carrier](/Users/xangma/repos/torchwave/src/torchwave/approximants/imrphenomp.py:317) and switch simplified angle terms. The “full two-spin” claim is misleading. XPHM builds on the simplified HM implementation. The published X family has explicit single-spin PN and double-spin MSA alternatives whose versions and conventions must be matched. [Precessing X paper](https://arxiv.org/abs/2004.06503) |
| NSBH | [The code divides `M*f_RD` by `2π*M_seconds` again](/Users/xangma/repos/torchwave/src/torchwave/approximants/imrphenom_nsbh.py:80), making ringdown frequency too small by 2π. It applies exponential windows to D_NRTidalv2 and does not calculate the claimed remnant torus attenuation. Implement the specified model, including disruption regimes. [PhenomNSBH paper](https://arxiv.org/abs/2002.08383) |
| ROM / SEOBNR | [Training uses `f=mf/(π*M_seconds)`](/Users/xangma/repos/torchwave/src/torchwave/approximants/rom/generator.py:155), while [evaluation uses `mf=M_seconds*f`](/Users/xangma/repos/torchwave/src/torchwave/approximants/rom/surrogate.py:136). This is an inconsistent frequency coordinate. Training on the local ODE validates fidelity to that ODE, not to LAL. v4/v5 aliases do not establish either model. |
| TEOBResumS | [The potential](/Users/xangma/repos/torchwave/src/torchwave/approximants/teobresums.py:163) contains only terms through u⁴ despite its 4PN/5PN description; [strain uses leading quadrupole amplitude](/Users/xangma/repos/torchwave/src/torchwave/approximants/teobresums.py:508). A differentiable EOB prototype is not evidence of TEOBResumS parity. |
| TaylorF2_HM / ringdown | The (2,2) internal consistency test does not certify higher modes. [TaylorF2_HM](/Users/xangma/repos/torchwave/src/torchwave/approximants/taylorf2_hm.py:93) supports (2,2), (2,1), (3,3), (4,4), not the claimed (3,2), and uses leading mode amplitudes. [The ringdown test](/Users/xangma/repos/torchwave/tests/test_ringdown.py:16) constructs its expected values from the same coefficient dictionary as the implementation; this does not independently validate the published table or LAL parity. |

The standalone continuous-wave, EMRI, memory, and ppE entries were not independently numerically audited here. They need their own declared references and validation; the supplied report does not establish equivalence to a LAL approximant for them.

Detection suitability, inference suitability, physical model accuracy, and reproduction of a specific LAL implementation are different claims. There is no demonstrated universal “LVK parity threshold” of 0.97–0.99; inference requirements depend on the application. [Waveform accuracy standards](https://arxiv.org/abs/0809.3844)

Clean-room headers are not a provenance record. Preserve the separation between independent mathematical implementation and binary-oracle validation, and trace each coefficient/table to an allowed publication or dataset. This review does not establish the provenance of all existing code or data. The [coding-agent prompt](/Users/xangma/repos/torchwave/docs/reviews/2026-09-10-parity/coding-agent-prompt.md) sets out the work and acceptance criteria.

Reproduce the probe from the repository:

```sh
PYTHONPATH=/Users/xangma/repos/torchwave/src /Users/xangma/miniconda3/envs/pycbcorig313/bin/python /Users/xangma/repos/torchwave/docs/reviews/2026-09-10-parity/probe.py
```

It rewrites `results.json` beside the script. The original selected-test command used a temporary pytest 9.1.1 installation (the existing oracle environment lacked pytest), with plugins and pytest caching disabled:

```sh
PYTHONPATH=/tmp/torchwave-parity-review-20260910/pytest-runner:/Users/xangma/repos/torchwave/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q -p no:cacheprovider tests/test_lal_parity.py tests/test_imrphenomp.py tests/test_taylorf2_spins.py tests/test_imrphenom_xas.py tests/test_imrphenom_hm.py tests/test_imrphenom_xphm.py tests/test_nrtidal.py -k lal --tb=short -rA
```
