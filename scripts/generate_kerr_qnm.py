"""Compatibility command for TorchWave's independently derived Kerr generator.

Run with --output-dir PATH. No external QNM solver or frequency table is used.
The original external-solver validation fixture is retained only for checking
results; it is not a calibration input to this generator.
"""

from generate_kerr_qnm_independent import main


if __name__ == "__main__":
    main()
