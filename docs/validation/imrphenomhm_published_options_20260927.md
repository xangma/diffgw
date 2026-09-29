# Published HM: source options and reference-frequency diagnostic

This exploratory study localizes two contributions to the failed published-HM
NR comparisons: the outer mapping transitions materially change the result,
and the current finite-reference adapter changes the physical template family.
Neither observation establishes a complete repair. The prospective
[2374/2163 results](imrphenomhm_published_nr_receipt_20260927.md) remain red.
No production formula, coefficient, reference version, or acceptance gate changed.

## Controlled comparisons

Every variant uses the same raw data, conditioned arrays, PSD, intrinsic
parameters, two views and four optimizer seeds as the frozen protocol.
The baseline is `calibration=published`, `hm_21_pn_variant=mishra_eq12b`,
`hm_32_pn_variant=mishra_eq12d`, `hm_43_pn_variant=mishra_eq12g`,
`hm_mapping_transitions=london_2018` and a 30 Hz reference. The binary PN
variants select `binary_compatible`; the runtime mapping selects
`phenomd_runtime`. Each individual option changes alone;
the combined variants are explicitly named. Both frontends are evaluated.
Results at the baseline alignment are recorded separately from reoptimized
results. These already-seen cases are diagnostic data, not new holdouts.

Second-view best-seed results (Torch shown; JAX gives the same conclusions):

| Variant | 2374 match / norm | 2163 match / norm |
| --- | ---: | ---: |
| Published baseline | 0.971607892 / 0.750351161 | 0.972272538 / 1.223703469 |
| Only inherited calibration | 0.971625675 / 0.750081746 | 0.972298278 / 1.226218191 |
| Only binary PN 21 | 0.971607892 / 0.750351161 | 0.972357225 / 1.224368008 |
| Only binary PN 32 | 0.971607892 / 0.750351161 | 0.974493240 / 1.254191331 |
| Only binary PN 43 | 0.971607892 / 0.750351161 | 0.972173655 / 1.221945332 |
| Only runtime mapping | 0.979449300 / 0.819294042 | 0.983465574 / 1.012288962 |
| All inherited defaults | 0.979523959 / 0.826973845 | 0.984399471 / 1.033534726 |
| Published calibration + binary PN + runtime map | 0.979449300 / 0.819294042 | 0.984409077 / 1.033481665 |
| Only reference=None | 0.971789397 / 0.811096799 | 0.977392313 / 0.824195979 |
| Only reference=40 Hz | 0.973583389 / 0.989988531 | 0.977446799 / 0.828646845 |

Only switching the mapping brings the spinning case within the numerical
match/norm bounds on this seen-data comparison. No tested option brings both
cases within both second-view bounds. Changing calibration alone shifts these
matches by less than 0.000026. PN changes vanish for the nonspinning case.

The `reference=None` variant has four-seed match spreads of 0.000104 and
0.000546 in the two 2374 views, above the original stability bound; two seeds
terminate at the orbital-angle interval boundary. The table reports the best
observed value, not a certified global maximum. All other option scans have
spreads below 2.01e-10. Baseline reproduction is exact for the retained metrics,
and maximum frontend relative complex mode L2 is below 1.50e-12.

## What the scalar residual establishes

At the failing view, replacing full NR with only the six modeled mode pairs
still gives model/NR match and norm ratio 0.976521/0.758241 for 2374 and
0.974253/1.222200 for 2163 at the original alignment. The corresponding
six-pair/full-NR matches are 0.997359 and 0.998830, with norm ratios
0.989594 and 1.001230. Missing modes alone therefore do not explain the
frozen failures. This calculation uses the actual scalar detector projection;
it is distinct from the earlier plus/cross network diagnostic.

The exact decomposition is

$$
|c-r|^2=(|c|-|r|)^2+4|c||r|\sin^2[\arg(cr^*)/2].
$$

Integrating with the frozen PSD and dividing by NR power gives:

| Case, second view | Total residual power | Magnitude term | Phase term |
| --- | ---: | ---: | ---: |
| 2374 | 0.104932645 | 0.085801605 | 0.019131041 |
| 2163 | 0.117903624 | 0.077662353 | 0.040241271 |

Magnitude terms account for about 82% and 66%. These are differences between
projected, aligned detector spectra, not a decomposition of intrinsic
amplitude-model error. The chosen orbital phase and polarization affect modal
interference. Pair residual powers require their cross terms; their diagonal
powers cannot be treated as additive attributions. The historical diagnostic
verified the Gram matrices' coherent sums; those raw matrices are excluded
from the distributed repository.

## Reference-frequency algebra

Let $r=M_s f_{\rm ref}$, $s$ be the independently evaluated D peak slope,
$B=sr$ and $\phi$ the raw joined D phase. The documented
[D conventions](../math/imrphenomd-waveform.md) give a constant carrier-phase
change

$$
C(r)=\Psi_{\rm ref}-\Psi_{\rm coal}=\pi/4-\phi(r)+B.
$$

The continuous London phase mapping takes a carrier constant $C$ to
$mC/2$ on every branch: continuity offsets telescope across the joins.
Combining this with the current explicit HM adapter yields

$$
\frac{P_{\ell m}(r)}{P_{\ell m}(\mathrm{None})}
=\exp[-imC(r)/2+i(m/2-1)B]
=\exp\{im[\phi(r)-\pi/4]/2\}\exp(-iB).
$$

This is an algebraic consequence of our documented composition, not an
NR-fitted phase correction. It applies on nonzero support with all mapped
carrier phase evaluations below the D hard cutoff. The orbital part can be
absorbed in one orbital rotation. The remaining common analytic Fourier phase
is generally not a physical polarization rotation: a projected mode's
coefficient has the form $a_{\ell m}\cos2\psi-i b_{\ell m}\sin2\psi$,
an ellipse rather than an arbitrary unit phasor. Circular polarization is a
special degeneracy. See the primary orbital rotation law in
[García-Quirós et al., Eq. (14)](https://arxiv.org/html/2001.10914v2);
that later model does not itself specify the HM reference-frequency API.

Both frontends, all six modes, both cases and the None→30, None→40 and
30→40 changes satisfy the independently predicted identity. The largest
supported-bin complex-ratio residual is 5.28e-13; the largest projected
residual against the common-phase prediction is 2.71e-13. The independent
60-digit scalar slope gives $B_{30}=6.707893462908468$ for 2374 and
$4.959913281018781$ for 2163. No value was inferred from NR.

A second diagnostic removes only $\exp[i(m/2-1)B]$ from the complete native
output. Both 30 and 40 Hz variants then reproduce the `None` detector spectrum
after the analytically predicted orbital compensation, with maximum
PSD-weighted relative complex L2 of 2.82e-13 over the full comparison band.
No optimizer is used to select that compensation; $(2,2)$ is unchanged.
Reoptimized second-view matches remain 0.971789397 and 0.977392313, with norm
ratios about 0.811100 and 0.824196. Both remain below the original match and
norm floors. Small norm differences between equivalent reoptimized variants
reflect the finite optimizer tolerance; compensated fixed-alignment spectra
agree at the quoted numerical precision. The `None` seed instability above
also prevents calling its best score a certified global optimum.

If `f_ref` is intended only to choose the orbital phase origin, this surviving
factor violates that intended covariance. Removing only the explicit HM factor
would leave $\exp[-imC/2]$, a pure orbital transformation, and preserve the
$(2,2)$ carrier. That is a mathematically motivated candidate API change; it
must be separately versioned and reviewed before production adoption.
Agreement with the current SymPy/mpmath composition verifies its implementation,
not the physical adequacy of this API convention.

## Sources, limits and next action

[London et al., Eqs. (3)–(9)](https://arxiv.org/html/1708.00404v4)
use the magnitude of the PN amplitude ratio and print the outer transition
constants 0.018 for amplitude and 0.014 for phase. Switching to the runtime
pair is a distinct model option, not a reason to revise those equations.
The paper also offers an alternative amplitude ratio after Eq. (4); it was
not tested here. The exact named Mishra polynomials define this repository's
option; London's wording about PN order alone does not prove that this tuple
reconstructs every choice in the historical implementation.

The next source task is to specify and test a reference-frequency convention
that preserves the intended orbital covariance and the $(2,2)$ reduction.
Residual physical disagreement still needs component-level investigation;
a higher score on these cases cannot choose a production coefficient or
retroactively qualify them. Any revised model requires a new reference version,
frontend checks and separately frozen untouched physical validation.
No external implementation source was inspected or transcribed.

## Historical evidence and maintained reproduction

This report retains the scoped conclusions, option tuples, selected metrics
and artifact identities of the exploratory study. The source options, raw
data and fixture hashes were verified against the unchanged frozen runner
before evaluation. Both backends and seeds `17,23,41,67` were checked using
population size 12, 120 iterations, tolerance `1e-9` and polishing. Every
residual decomposition and coherent-sum identity was checked, and baseline
scores were reproduced. Full seed traces, matrices and diagnostic arrays are
excluded from the distributed repository.

All numerical work completed on `len` using the original PyCBC-compatible
Python environment. The native source revision is
`4449a7340a7198b25a29c662c2030b91f001b002`; the maintained baseline metrics are
from `248ff5f`. Python
3.12.3, NumPy 2.3.5, SciPy 1.16.3, PyCBC 2.11.0, Torch 2.14.0,
JAX/JAXlib 0.11.2, SymPy 1.14.0 and mpmath 1.3.0 were used on CPU in FP64.

The exploratory execution used
`len:/tmp/diffgw-hm-published-options-20260927/`. The two job wrappers exited 0;
scientific qualification remains false. These historical scratch scripts are
not shipped or promised to remain available. Their recorded identities are:

| Script | SHA-256 |
| --- | --- |
| `hm_published_option_diagnostic.py` | `2309fa8bcb1e309edd633a0415f18a9a33a899561e501f73bd6b8b02b8c14f53` |
| `hm_published_projected_residual.py` | `4eb1968e394bafcdcfae4bfd35422d80d14f87595a407a54f2d038b4c0c519be` |
| `hm_published_gauge_identity.py` | `0a98dab3996cc97cfbfb1ac4cf60119042a5964b8585d7f11bfa1fc05b46c213` |
| `hm_published_covariance_diagnostic.py` | `ccee2b406f61636f08db387ec8a939ffd6411f24553309d03eddd84fd93805f6` |

The recorded case-specific SHA-256 identities are:

| Artifact | 2374 | 2163 |
| --- | --- | --- |
| Frozen manifest | `4381ddf5537e328c0708ce9f77152ea985c075302a5d7f1f09e62fd2cb13fccd` | `477083ba0fa8ba6346008d946cd41504abc1111d9f727e82ff8bdbc42ed80b58` |
| Fixture | `e163d466ac0f56f6506d09ce438b4c52cd919fefab7197884038e038f085d0b1` | `bb6cc4dc43f726aca4c9209c76d7ba615167aaa89e9fe06e6bbe841a366ae707` |
| Baseline metrics | `f3892398509fe687e148911566277d65e022f2be9fc6473018c96ed311ccf73f` | `74e9b7c665baa341fcaed68a768947f575cb994a8304723113fe496ccdfcd23e` |
| Full option output | `0ff3524430f67b5ac324dc7fa1029d71ba58d77fad72242021b11b497f53b32a` | `25ee9696f303abefa239c1373e884f465ac372963ea3dd889e369176046e39e6` |
| Full covariance output | `a6a6f0a8d372e6a5f9f9df382860a06f23714f69052fd46f7ff5d75ef98d2a1f` | `1cd7a0299730d93abe6697076c136b4e638292b6df16eca76cfab04b1ca07f36` |

| Shared artifact | SHA-256 |
| --- | --- |
| Frozen runner | `1cba5bb618c1994cf4851c6dbcf3a2a7c46b95d9ccc3706805432b5e297c6be2` |
| Base runner | `89856b9b94af8cf546620d2f07928016a35f8ddafa4277e9c479b0638d18b2c0` |
| Source Python tree | `15d843ade209bbd36a65b8c1b850a415182d4296ab3994dd1dcdd304a9cd446a` |
| Full residual output | `5a092253c053f4d468a64a619f774b3a929d9ab32e9855c6c508f95410a57db2` |
| Full gauge-identity output | `6f3fb8a1f71388eb99b6d0081c8f49dab8cfa516d6a510b0aa8aa10b9149e4f5` |
| Removed combined diagnostic JSON | `6f9b720937b9ee789c61601d57451227efe870631c0a5aa1d45527915edfe53d` |

These hashes identify historical evidence; they do not make the exploratory
study replayable from the distributed repository. The separately frozen
[maintained NR protocol](imrphenomhm_published_nr_protocol_20260927.md)
retains its runners, input identities and reproduction commands, and the
[prospective receipts](imrphenomhm_published_nr_receipt_20260927.md) retain
their failed gates.

The generated waveform-library documentation check and strict MkDocs build
passed on len after integrating this evidence. No source or test code changed.
