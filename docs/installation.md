# Installation

Install DiffGW in editable mode with the frontend and optional tools you need:

```bash
git clone https://github.com/xangma/diffgw.git
cd diffgw

# PyTorch only
pip install -e ".[torch]"

# JAX only
pip install -e ".[jax]"

# All backends, inference tools, and documentation dependencies
pip install -e ".[all,dev,docs]"
```

Optional ROM datasets are acquired separately; see [External Datasets](external-data.md).
After installation, continue with the [quickstart tutorial](tutorials/01_quickstart_and_parity.md).
