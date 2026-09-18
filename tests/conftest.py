import os

# Prevent macOS crashes when conda-forge OpenBLAS and PyTorch bundle separate libomp runtimes
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
