# (Experimental) Python bindings

LLZK has experimental support for MLIR's Python bindings.

Prerequisites:
* The Python packages required for MLIR's Python bindings must be installed, as
  indicated in the `mlir/python/requirements.txt` file in the LLVM monorepo's
  source tree.
* You must build and link LLZK against a version of MLIR built with
  `MLIR_ENABLE_BINDINGS_PYTHON` set to `ON`. In the Nix setup, this can be
  accessed using the `llzkWithPython` output.
* LLZK must be configured with `-DLLZK_ENABLE_BINDINGS_PYTHON=ON`.
