# Pass 19 KerrRingdown independent review

Decision: **RETAIN** the current native behavior. I found no source supported
runtime correction. The implementation is a defined restricted ringdown
primitive, not a complete merger excitation or physical strain model.

## Scope

I compared the primary review with the directly relevant native source
`src/torchwave/approximants/kerr_ringdown.py`, its direct ringdown tests, and
only the cited primary papers: Berti, Cardoso & Will,
[arXiv:gr-qc/0512160v2](https://arxiv.org/abs/gr-qc/0512160v2), Berti & Cardoso,
[arXiv:gr-qc/0605118v2](https://arxiv.org/abs/gr-qc/0605118v2), and Cook & Zalutskiy,
[arXiv:1410.7698v2](https://arxiv.org/abs/1410.7698v2).

## Findings

- **Mass, distance, and redshift:** The primary waveform carries `M/D_L`,
  while QNM frequency and damping use dimensionless `M*omega` and
  `M*gamma`, with detector-frame redshifted mass. The native code correctly
  converts solar masses to seconds, returns ω in rad/s and τ in seconds, and
  forms `M*(omega-i/tau)`. It has no distance argument, so its amplitude is
  a supplied native modal coefficient; adding (M/D_L) would change API
  semantics.
- **Twin modes and polarization:** BCW Eq. (2.9) has two independent real
  amplitude/phase pairs for each positive-(m) label, related to the
  opposite-(m), opposite-frequency solution. The native API supplies one
  amplitude and phase and uses an equal-twin projection. That is a valid
  restricted convention, but cannot represent arbitrary polarization. No
  correction is safe without a new twin-mode interface or metadata.
- **Overtones:** The excitation paper gives `C_n = B_n` times an initial-data
  source integral. Frequencies and universal excitation factors therefore do
  not determine amplitudes, phases, or overtone weights. The native API accepts
  arbitrary ((l,m,n)) amplitudes and phases and defaults to (n=0); no paper
  supports silently choosing a different generic-merger overtone model.
- **Complex spheroidal factors:** Cook & Zalutskiy fix the azimuthal factor,
  phase continuation from `c=0`, normalization, and the `m<->-m` symmetry.
  The opt-in native spheroidal path evaluates the complex argument
  `c=(chi_f*M)*(omega-i/tau)` and constructs the equatorial partner from
  the same positive-(m) eigensolution. The spherical default is an explicit
  approximation and changing it would alter established outputs.
- **Fourier convention:** For the documented transform
  `integral h(t)*exp(-2*pi*i*f*t) dt`, the real causal exponential has both
  `1/[gamma+i*(2*pi*f-omega)]` and
  `1/[gamma+i*(2*pi*f+omega)]` poles, multiplied by
  `exp(-2*pi*i*f*t0)`. The native frequency path implements exactly this
  two-pole expression, including the one-half real-signal factor.

## Verdict

**RETAIN.** The direct tests cover the stated causal time/Fourier pair,
Hermitian and onset-shift behavior, mass scaling, spherical/spheroidal limits,
overtone inputs, batching, and gradients. The only safe follow-up is
documentation of the restricted equal-twin, amplitude-supplied convention and
its missing (D_L)/independent-twin metadata; no runtime or test change is
justified by the cited sources.
