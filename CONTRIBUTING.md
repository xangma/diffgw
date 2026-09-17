# Contributing to diffgw (torchwave & jaxwave)

Thank you for your interest in contributing to **diffgw**! diffgw provides native, autograd-differentiable gravitational-wave modeling in both PyTorch (`torchwave`) and JAX (`jaxwave`), powered by a zero-dependency analytical core (`diffgw_core`).

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
   - diffgw is an independent, native Python/PyTorch/JAX framework. All contributions must be original code rather than adapted or copied from copyleft/GPL codebases.
   - Standard reference tools are welcome as external benchmarks and validation oracles for unit test tolerances.
3. **Citation & Provenance Documentation**:
   - Every new approximant or calibration must cite its foundational paper in [`CITATIONS.md`](CITATIONS.md).
   - Document derivations, coefficient sources, and sign conventions in [`docs/provenance_ledger.md`](docs/provenance_ledger.md) and [`docs/differences_from_lal.md`](docs/differences_from_lal.md).
4. **Third-Party Data & Calibration Assets**:
   - External calibration datasets (such as Numerical Relativity waveforms or public surrogate meshes) must have documented license terms compatible with distribution or external download.
   - Record all dataset attributions in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

---

## 3. Cross-Framework FP64 Parity Requirement

A defining feature of diffgw is complete cross-framework parity between PyTorch (`torchwave`) and JAX (`jaxwave`).

### Parity Standards
- **Co-Equal Implementations**: Any new waveform approximant, detector projection, or inference tool must be implemented in **both** `torchwave` (PyTorch) and `jaxwave` (JAX), sharing common analytical physics constants and coefficient tables via `diffgw_core`.
- **FP64 Precision**: All waveform computations enforce double precision (`torch.float64` in PyTorch, `jax.config.update("jax_enable_x64", True)` in JAX).
- **Mathematical Agreement**:
  - Waveform evaluations across both frontends must match to double-precision numerical tolerance:
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
