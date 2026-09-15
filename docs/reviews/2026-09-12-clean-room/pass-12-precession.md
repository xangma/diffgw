# Pass 12: independent remnant selection and source audit

XP and XPHM now accept `final_spin_mod=0`, selecting the existing paper-derived
chi_p remnant estimate independently of the Euler-angle version. The override
reaches XAS/XHM, including when MSA angles are selected. Omitting the keyword
preserves earlier behavior: NNLO versions 101--104 select this remnant; other
versions retain the aligned carrier remnant. Other explicit remnant options
raise an error. This exposes one supported prescription, not full MSA parity.

Pratten et al., [arXiv:2004.06503v2](https://arxiv.org/abs/2004.06503v2),
Table V and Appendix F describe the independent final-spin selector. Table III
and the numerical-relativity injection discussion describe the 223/3/convention1 published default. TorchWave's
established `nnlo` default resolves to 102; explicit `msa` resolves to 223.
Changing the angle default alone would leave the default MSA remnant missing.

The Luna candidate attempted to integrate FinalSpinMod3 using the printed
initial-mass denominator in Eq. 4.29 with the final-mass angular momentum in
Eq. 4.18. Parent review rejected that combination: its aligned limit yields
m_final^2 times the aligned Kerr spin. An independently justified normalized,
signed remnant and degenerate-state handling are still needed. No experimental
Mod3 integration remains. Pass 11's averaged-coupling and squared-angular-
momentum helpers are unchanged.

Colleoni et al., [arXiv:2412.16721v1](https://arxiv.org/abs/2412.16721v1),
Sec. II.2, describes a separate SpinTaylor prescription selected with 320/2.
It does not resolve the MSA mass-denominator question and is not mixed into the
present NNLO/MSA equations.

New tests check version defaults and unsupported remnant rejection. Production
carrier interception verifies that explicit Mod0 reaches both models with
MSA angles, while default NNLO output stays bitwise identical. The underlying
remnant equation retains pass-10 independent equation and gradient tests.
See the pass-12 summary for final verification.

Primary TeX SHA256:
`d1e611c582b2861bc34d975e99b6c2d6ef3d4447442aab249541616a331443f5`.
The published PDF hash and exact source scope are recorded in source-audit.json.
Luna used primary papers and inherited native code, with no LAL/PyCBC
implementation source or parent binary residuals provided.
