# Contributing to diffgw (torchwave & jaxwave)

Thank you for your interest in contributing to **diffgw**! diffgw provides native, autograd-differentiable gravitational-wave modeling in both PyTorch (`torchwave`) and JAX (`jaxwave`), powered by a framework-independent analytical core (`diffgw_core`).

We welcome contributions ranging from new waveform approximants and detector networks to bug fixes, performance optimizations, and documentation improvements.

---

## 1. Code of Conduct & Community Tone

diffgw is an open scientific project committed to providing a welcoming, professional, and harassment-free environment for everyone.

### Our Values
- **Respect & Inclusivity**: Treat all contributors, maintainers, and users with respect, regardless of background, identity, or experience level.
- **Scientific Rigor & Constructive Discourse**: Critique ideas, mathematics, and implementations constructively. Frame discussions around evidence, peer-reviewed literature, and empirical benchmark results.
- **Collaboration & Mentorship**: Support newcomers, provide clear feedback during code reviews, and foster open scientific collaboration.

Unacceptable behavior—including harassment, insults, exclusionary language, or unprofessional conduct—will not be tolerated in issues, pull requests, or project forums.

---

## 2. Licensing & Implementation Guidelines

diffgw is released under the permissive **MIT License**. To ensure diffgw remains independent, lightweight, and strictly MIT-licensed, contributions must follow these guidelines:

1. **Literature-First Derivations**:
   - All physics models, waveform phasing expressions, ringdown fits, and coordinate transformations should be derived directly from primary **peer-reviewed literature**, public technical reports, or independent first-principles derivations.
2. **Original Implementations**:
   - diffgw is an independent, native Python/PyTorch/JAX framework. Contributions must not copy or adapt copyleft/GPL implementation code. Permissively licensed adaptations require explicit source, version, license, and attribution records.
   - Standard reference tools are welcome as external benchmarks and validation oracles for unit test tolerances.
3. **Citation & Provenance Documentation**:
   - Every new approximant or calibration must cite its foundational paper in [`CITATIONS.md`](CITATIONS.md).
   - Document derivations, coefficient sources, and sign conventions in [`docs/provenance_ledger.md`](docs/provenance_ledger.md) and [`docs/comparison.md`](docs/comparison.md#model-specific-differences-from-lal).
4. **Third-Party Data & Calibration Assets**:
   - External calibration datasets (such as Numerical Relativity waveforms or public surrogate meshes) must have documented license terms compatible with distribution or external download.
   - Record all dataset attributions in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

---

## 3. Cross-Framework FP64 Parity Requirement

### Prototype and release boundary

Unvalidated waveform models, physics components, surrogate tables, and training experiments
belong on a development or feature branch until their declared scientific gates
pass. Keep research under `dev/` on that branch; do not merge it into `main`,
register public aliases, export it from `src/`, or include it in distributions.
Tested mathematical components may land separately with their bounded claims.

Before promoting a native ROM, qualify its source generator, held-out compression
and interpolation errors, both runtime frontends, and parameter derivatives over
a declared domain. Record source and data hashes, numerical settings, independent
references, thresholds and failures. Finite output and backend agreement alone
do not validate the physics. Packaging tests enforce the development boundary.

Contributions should maintain numerical consistency between PyTorch (`torchwave`) and JAX (`jaxwave`). The standards below are development targets; current coverage and model-specific exceptions are recorded in the [validation guide](docs/validation.md).

### Parity Standards
- **Co-Equal Implementations**: Any new waveform approximant, detector projection, or inference tool must be implemented in **both** `torchwave` (PyTorch) and `jaxwave` (JAX), sharing common analytical physics constants and coefficient tables via `diffgw_core`.
- **FP64 Precision**: Reference checks must use double precision (`torch.float64` in PyTorch, `jax.config.update("jax_enable_x64", True)` in JAX).
- **Mathematical Agreement**:
  - Target the following cross-frontend tolerances, documenting any justified model-specific exceptions and metric definitions:
    $$\text{Relative } L_2 \text{ error} \le 10^{-11}, \quad \text{Mismatch } 1 - \mathcal{O}(h_{\text{torch}}, h_{\text{jax}}) \le 10^{-14}$$
  - Parity tests must cover polarizations ($h_+$, $h_\times$), individual multipoles $(\ell, m)$, phase derivatives, and edge parameter regimes (e.g., extreme mass ratios, near-extremal spins).
- **Mandatory Parity Tests**: Add a dedicated parity test under `tests/parity/test_<model>_parity.py` for each new model or modification.

---

## 4. Development Instructions

### Setting Up the Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/xangma/diffgw.git
   cd diffgw
   ```

2. Create and activate a Python 3.10+ virtual environment (conda or venv):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install in editable mode with development and backend dependencies:
   ```bash
   # Full installation (PyTorch, JAX, inference tools, and dev dependencies)
   pip install -e ".[all,dev]"
   ```

   Alternatively, install specific backend targets:
   ```bash
   pip install -e ".[torch,dev]"  # PyTorch only
   pip install -e ".[jax,dev]"    # JAX only
   ```

### Running Tests

We use `pytest` for automated unit, integration, and parity testing.

- **Run fast unit tests**:
  ```bash
  pytest tests/ -k "not parity"
  ```

- **Run cross-framework parity tests**:
  ```bash
  pytest tests/parity/ -v
  ```

- **Run a specific test file**:
  ```bash
  pytest tests/parity/test_phenomd_parity.py -v
  ```

- **Run full test suite**:
  ```bash
  pytest
  ```

### Code Formatting and Linting

We enforce code style and formatting using `ruff`.

- **Check formatting without modifying files**:
  ```bash
  ruff format --check src/ tests/
  ```

- **Automatically format code**:
  ```bash
  ruff format src/ tests/
  ```

- **Run linter**:
  ```bash
  ruff check src/ tests/
  ```

---

## 5. Pull Request Guidelines

Before submitting a pull request:
1. Ensure all existing and new tests pass (`pytest`).
2. Verify FP64 parity between PyTorch and JAX if adding or modifying physical models.
3. Ensure formatting conforms to project standards (`ruff format` and `ruff check`).
4. Update [`CITATIONS.md`](CITATIONS.md) and [`docs/provenance_ledger.md`](docs/provenance_ledger.md) with relevant academic references and mathematical derivations.
5. Provide a clear PR description explaining the scientific motivation, mathematical derivation, and verification benchmarks.


## 6. Documentation Evidence

Quantitative claims must link to committed measurements or a reproducible test and state their scope. A test threshold is an acceptance criterion, not a measured outcome. For each performance comparison, identify the baseline, source versions, model/options, parameter set, frequency/time grid, dtype, device, batch size, timing boundary, synchronization, repetitions, and uncertainty. Keep failed numerical checks visible and separate diagnostic timings from qualified results.

Distinguish interface availability, frontend consistency, external-reference agreement, derivative checks, and independent physical validation. Numerical tolerances are not exact equality or parameter-domain guarantees. A method paper supports the algorithm description; its performance numbers do not establish DiffGW's performance. Use [validation evidence](docs/validation.md) and the [benchmark protocol](docs/benchmarking_protocol.md) when writing or updating claims.
