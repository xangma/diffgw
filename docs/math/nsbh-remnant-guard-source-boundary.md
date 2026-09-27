# NSBH remnant guard source boundary

## Scope and status

`nsbh.remnant.guards/v0` is **blocked**, not an executable paper-alignment
gate. This audit separates the published remnant fit and physical bounds from
the extra clipping/fallback rules in the native Torch/JAX providers. The
[printed-precision interval gate](nsbh-remnant-rounding-interval.md) remains
valid on its five guard-inactive cases; this page does not change its domain,
tolerance, or status. No whole-waveform claim follows from this audit.

## Inputs, conventions, and options

Write $q=m_{\rm BH}/m_{\rm NS}$, $\nu=q/(1+q)^2$,
$a=a_{\rm BH}$, and $\Lambda=\Lambda_{\rm NS}$. Set the NS spin to zero and
$m_{\rm NS}=1.4\,M_\odot$. The diagnostic inputs are
$(q,a,\Lambda)=(3,0,100)$ and $(5,-0.5,100)$. Both lie within the reported
individual parameter ranges $\nu\in[0.109,0.222]$,
$a\in[-0.5,0.75]$, and $\Lambda\in[100,2500]$; this does not assert that
either tuple is an NR calibration datum.

## Mathematics and components

[Zappa et al. (2019), main-text Eqs. (1)–(2) and Supplementary Eqs.
(4)–(6), Table 1, §I](https://arxiv.org/html/1903.11622v2) define
$X_\bullet=M_\bullet/M$ and $a_\bullet$, then multiply the corresponding
Jiménez-Forteza BBH quantity by a rational tidal factor $G_F$. The paper
describes $X_\bullet>1$ and $a_\bullet<-1$ as unphysical extrapolation
outcomes and says it forces agreement with BBH, while also writing
“$X_\bullet=1$ and $|a_\bullet|\leq1$.” Those statements do not give a
unique pointwise prescription: setting the *total* mass ratio to one and
setting the *tidal factor* to one differ whenever $X_{\rm BBH}<1$. The text
does not state a numerical switch at $\nu=0.188$, a factor-above-one clip,
or equality/tie conventions.

The independent exact-rational [scalar reference](../../src/diffgw_reference/nsbh_remnant_rounding.py)
propagates half-last-printed-digit coefficient intervals through the published
equations. The following bounds are rounded **outward** for display; the
underlying interval evaluation uses exact fractions:

| $(q,a,\Lambda)$ | $G_M$ | $G_a$ | Unclipped $X_\bullet$ |
| --- | ---: | ---: | ---: |
| $(3,0,100)$ | $(1.000052,1.002250)$ | $(1.013955,1.017116)$ | $(0.971295,0.973441)$ |
| $(5,-0.5,100)$ | $(1.005855,1.007315)$ | $(1.012851,1.015142)$ | $(0.992122,0.993630)$ |

Both tidal factors are unambiguously above one in these examples, yet the
*total* mass ratio remains below one. The source also explicitly describes
the small-$\Lambda$ increase of nonspinning NSBH mass and spin relative to
BBH in its main text. Thus $G>1$ alone is not the paper's stated unphysical
condition. The [Jiménez-Forteza et al. (2017)](https://arxiv.org/html/1611.00332v2)
unity spin cap is stated for its **precessing augmented** fit (Eq. (20)); it
does not determine a guard for the aligned-spin BBH baseline used here.

## Implementation mapping

Each native `zappa_remnant_properties` in
`src/torchwave/approximants/imrphenom_nsbh.py` and
`src/jaxwave/approximants/imrphenom_nsbh.py` independently sets a mass or
spin **factor** above one to one. Both also set both factors to one when
`(a < 0 and nu < 0.188) or a < -0.5`. For $(3,0,100)$, factor clipping is
active although the unclipped total quantities remain physical. For
$(5,-0.5,100)$, factor clipping and the `nu < 0.188` fallback are active;
$a=-0.5$ does not satisfy the separate strict `a < -0.5` test. These are
descriptions of native behavior, not source-derived rules. This audit does
not change either provider.

## Evidence and limitations

The cited paper supplies fit coefficients and broad physical constraints but
not the predicates or output substitutions needed for a unique scalar guard
reference. In particular, the paper's “agree with BBH” and
“$X_\bullet=1$” formulations are unequal, and it does not identify the
native $\nu=0.188$ threshold. A gate that compared the native guarded
outputs to a scalar transcription of those native rules would be circular;
a gate that imposed one interpretation of the paper would invent a rule.
The exact printed-coefficient intervals above diagnose this gap without
inferring missing coefficient digits or using LAL source/binary output.
Resolving the guard policy requires an explicit author prescription or a
new project policy labeled separately from paper qualification. NR accuracy,
negative-spin physical bounds over a wider domain, derivatives, and
whole-waveform assembly remain unqualified.

## Reproduce the checks

From the repository root, run
`PYTHONPATH=src python3 -m pytest -q tests/test_nsbh_remnant_guard_source_boundary.py`
for the source-only diagnostic. The test uses `bbh_remnant_interval` and
`zappa_factor_interval` with the two tuples above and asserts that both
printed-coefficient factors exceed one while the full remnant values remain
physical; it makes no native guarded-output comparison. The existing
**guard-inactive** gate is
`PYTHONPATH=src python3 -m pytest -q tests/test_nsbh_remnant_rounding_math_reference.py`
where Torch/JAX are installed. Check the six-section catalog with
`PYTHONPATH=src python3 scripts/generate_waveform_library_table.py --check`.
