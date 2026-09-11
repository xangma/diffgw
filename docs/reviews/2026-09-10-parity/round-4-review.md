# Fourth independent parity review

The latest response makes useful implementation progress but does not complete the parity objective. No waveform implementation files were edited by this review.

Reproduced on the existing CPU FP64 oracle environment: **215 tests passed in 14.79 s**, and the harness explicitly invoked with `--strict` passed its six TaylorF2/TaylorF2Ecc cases. Existing negative controls now reject bad DC, bad PSD mismatch, doubled test projection and missing PyCBC in strict CLI comparisons. Wrapper alias repairs and the improved phase derivative check are present.

Remaining material findings:

- **XAS is still a hybrid.** Its amplitude uses some public XAS fits; its entire phase is exactly D’s returned phase. Phase and intermediate-amplitude fit functions remain unused. Its four advertised cases have magnitude relative L2 errors of 0.0304–0.0793. Phase/time alignment cannot remove a magnitude discrepancy.
- **XHM remains HM’s empty subclass.** Identical parameters give identical TorchWave outputs. An inclined `(40,20,.3,-.1)` case has plus optimized match 0.98357 against XHM’s oracle and 0.96666 against HM’s oracle. Neither supports the new verified-parity claim.
- **New default-spin regression.** XPHM raises `TypeError: must be real number, not object` when spins are omitted, because raw sentinel arguments overwrite resolved values. The documented direct eccentric spin alias also mistakes an omitted value for conflicting explicit zero.
- **Strict false passes remain.** The added controls demonstrate passes for corrupted lower/upper out-of-band bins, NaN optimized matches, an empty case inventory, and missing PyCBC through pytest despite `STRICT_PARITY=1`. Shared incorrect detector geometry passes because both waveforms use the same projection; detector time delays are absent from that comparison.
- **D improved, but is unfinished.** Its high-spin magnitude error fell from the earlier approximately 0.014 to 0.0043086, with norm ratio 1.0001504 and optimized match 0.9999744. The report carries stale numbers and an unproven causal explanation.

The original requested engineering gates remain complex relative L2/norm error at most 1e-6 and required mismatches at most 1e-8. These are the agreed task criteria, not a universal scientific threshold. Diagnostic overlaps here optimize time/phase independently per polarization; raw complex residuals retain convention differences and are not used to diagnose physical model error.

Full configurations, controls, output logs and source identity are in `round-4-evidence/`. See `round-4-prompt.md` for the coding-agent continuation.
