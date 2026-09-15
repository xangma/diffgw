# Pass 34: XHM source-backed review

## Scope

This bounded pass reviewed the current `IMRPhenomXHM` runtime against the
primary IMRPhenomXHM paper (García-Quirós et al., arXiv:2001.10914v2) and the
existing clean-room pass-33 localization. No LAL, PyCBC, TEOBResumS, or other
waveform implementation source was inspected or used.

## Result

No new uniquely source-backed amplitude defect was found that is safe to patch
in this pass. The current 3,2 spheroidal ringdown phase contains one linear
`d_RD * f` term, matching the single term in Eq. (6.4); the earlier suspected
double-term issue is not present in the current worktree. The amplitude
ringdown factor matches Eq. (6.2): `f^(-1/12)`, `|a_lambda|`, the mode QNM
width, and the exponential factor. The transition frequencies, inverse
polynomial construction, odd-mode magnitude folding, and 3,2 spherical
mixing were reviewed without a source-complete contradiction beyond audits
already recorded in passes 17 and 33.

The remaining higher-mode frequency-dependent losses therefore remain
consistent with the unresolved calibration and phase/gauge metadata already
identified by pass 33. In particular, the paper does not provide enough
information to select between the surviving 3,2 transition-frequency
interpretations or to reconstruct the archived calibration/runtime gauge from
public equations alone. Applying an empirically chosen change here would
violate the clean-room and no-residual-fitting constraints.

## Validation

Focused XHM reconstruction, phase, amplitude, odd-amplitude, pole-selection,
and veto tests: **25 passed**.

`git diff --check`: passed.
