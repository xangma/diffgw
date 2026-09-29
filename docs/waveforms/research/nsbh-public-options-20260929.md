# IMRPhenomNSBH public options and evidence boundary (2026-09-29)

This comparison is pinned to the tested `len` Python environment:
LALSimulation **6.2.1**, PyCBC **2.11.0**, and the diffgw
`codex/nsbh-qualification` branch. Its purpose is to identify actual public
interfaces and option differences, not to infer equation identity from the
`IMRPhenomNSBH` name. The linked [LALSuite 7.26 API reference](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_n_s_b_h_8c.html)
describes the public model family. Its [driver page](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/group___l_a_l_sim_i_m_r_phenom__c.html)
identifies LALSimulation 6.2.1, matching the tested binary. The PyCBC link
is current online API documentation; the installed 2.11.0 interface was
also inspected directly.

| Option | Public LALSimulation / PyCBC | diffgw Torch and JAX |
| --- | --- | --- |
| Model and output | LAL's single-spin, nonprecessing FD `IMRPhenomNSBH` returns complex plus/cross series. The 6.2.1 binary resolves the approximant name; PyCBC 2.11.0 lists it among FD approximants. [LAL model reference](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_n_s_b_h_8c.html), [PyCBC waveform API](https://pycbc.org/pycbc/latest/html/pycbc.waveform.html). | Native Torch `IMRPhenomNSBH.forward` and JAX `IMRPhenomNSBH.evaluate` return complex plus/cross arrays. A backend match alone is not an independent source check. |
| Frequency selection | LAL offers a specified-frequency sequence and a uniform-$\Delta f$ driver; the latter zero-fills below $f_{\rm low}$ and beyond its ringdown cutoff. PyCBC documents both `get_fd_waveform` and `get_fd_waveform_sequence`. [LAL driver reference](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/group___l_a_l_sim_i_m_r_phenom__c.html), [PyCBC API](https://pycbc.org/pycbc/latest/html/pycbc.waveform.html). | Caller supplies a frequency tensor/array. The checked public path gives exact zero at nonpositive bins; this is not LAL's uniform-grid cutoff/zero-fill contract. |
| Physical parameters | LAL exposes BH/NS masses, aligned component spins, distance, reference phase/frequency and tidal parameters through `LALDict`; its 7.26 reference reports hard bounds $m_{\rm NS}\le3M_\odot$, $1\le q\le100$, $0\le\Lambda_2\le5000$, and warns if $\chi_{\rm NS}\ne0$. PyCBC's generic FD API includes `mass1`, `mass2`, `spin1z`, `spin2z`, `lambda2`, `distance`, `inclination`, and `f_ref`. [LAL domain](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_n_s_b_h_8c.html), [PyCBC parameters](https://pycbc.org/pycbc/latest/html/pycbc.waveform.html). | Public arguments include `mass1`, `mass2`, `chi1`, `chi2`, `lambda2`, `distance`, `inclination`, `phic`, `tc`, `f_ref`; the source-calibrated family assumes a nonspinning NS. Merely accepting `chi2` does not validate nonzero NS spin. The native path does not reproduce LAL's listed hard domain checks. |
| Phase and gradients | LAL documents an IMRPhenomD_NRTidalv2 phase and reference frequency. The PyCBC calls in this project's binary regressions are a LAL-backed oracle, not a second published derivation. [LAL model reference](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_n_s_b_h_8c.html). | Default phase has an inherited frequency-grid-dependent affine gauge; explicit `coalescence` and `reference_frequency` gauges are separately checked against a scalar PhenomD+NRTidalv2 reference. Both native backends support tensor differentiation in scoped tests. The default gauge is not established as a published LAL option. |

The [conditional complex assembly](../../math/nsbh-polarization-composition.md)
now verifies the last positive-frequency algebra for supplied carriers,
joining centers, scale and explicit phase. The public binary regression
tests fixed cases against the installed LAL/PyCBC stack, which is useful
compatibility evidence but cannot resolve the three
[printed-versus-native amplitude variants](../../math/nsbh-amplitude-variant-provenance.md)
or the [remnant fallback predicate](../../math/nsbh-remnant-guard-source-boundary.md).
Consequently neither this comparison nor the new component gate establishes
an option-equivalent, paper-derived whole-waveform implementation.
