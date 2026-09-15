# Independent reference coverage and model gaps

Updated after the continued reconstruction passes on 12 September 2026.
The strict harness now uses independent public waveform binaries for all
three EOB cases. Two model-level reference gaps remain: TaylorF2 higher
modes and the complete Kerr-ringdown waveform. Coverage does not imply parity.

| Model | Current independent measurement | Remaining work |
| --- | --- | --- |
| TaylorF2_HM | Its selected `(2,2)` carrier is compared with external TaylorF2. | Define and test all higher-mode PN orders, normalization, phase and mode content. Carrier agreement does not certify the full model. |
| KerrRingdown | Fifteen QNM families/overtones have fresh author-data tests; 60 public binary QNM cases and angular components are measured separately. | Resolve complete waveform amplitude, energy, phase and angular conventions. Component tests do not certify the full ringdown generator. |
| SEOBNR_ODE | Compared against the public `SEOBNRv4` waveform interface. | Finish a compatible calibrated model: complete radiation, resolved peak, NQC application, remnant mapping and merger attachment. |
| SEOBNR_ROM | Compared against public `SEOBNRv4_ROM`, using an independently downloaded and hash-pinned external data asset. | The native surrogate still models TorchWave's prototype generator; validate that generator and retrain a surrogate before claiming target equivalence. |
| TEOBResumS | Compared against public `TEOBResumS` through the installed LALSimulation interface. | Reconstruct that model's own dynamics, radiation, tidal physics and merger prescription. Shared EOB terminology does not establish equivalence to SEOBNRv4. |

The external data download is implemented by
[`scripts/fetch_lal_oracle_data.py`](../../../scripts/fetch_lal_oracle_data.py).
The exact oracle interface, versions, parameters, grid, raw metrics and status
are recorded in [the current catalog](verification/pass-3/catalog.json).
Missing data or an unavailable oracle causes an explicit failure or unresolved
status; native self-references are not substituted.

The default native EOB waveform remains a prototype. The continued pass
corrected Hamiltonian forces, batch differentiation, geometric units,
Hamiltonian angular frequency, physical strain projection and Fourier
measure. Its default radiation still contains only a short PN flux, and
its ringdown attaches one damped sinusoid at the retained trajectory endpoint.
The default surrogate inherits this incomplete training model.

New, explicit SEOBNR_ODE options provide a published conservative
Hamiltonian and factorized radiation: `conservative_model="seobnrv4_published"`,
`radiation_model="din2009_nonspinning"`, and
`radiation_model="pan2011_leading_spin"`. The latter two emit factorized `(2,2)`
strain and sum their specified multipolar fluxes, but omit the complete v4
spin terms, non-Keplerian prescription, NQC and calibrated merger attachment.
Their existence does not change the meaning of default parity measurements.
See [Hamiltonian](pass-3-seobnrv4.md) and
[factorized radiation](pass-3-factorized-modes.md).

Standalone [NQC equations](pass-3-eob-nqc.md) and
[merger-ringdown matching](pass-3-eob-merger-ringdown.md) implement additional
published components. They require resolved trajectory/matching inputs and
are not silently applied at an arbitrary endpoint.

The [SEOBNRv4 paper](https://arxiv.org/abs/1611.03703v3) describes separate
inspiral-plunge and merger-ringdown calibration; the
[SEOBNRv5HM paper](https://arxiv.org/abs/2303.18039v2) specifies further PN,
higher-mode and self-force information. Neither is implemented merely by
selecting an EOB Hamiltonian. External implementation source was excluded
from this reconstruction; inherited TorchWave provenance remains unverified.
