# Tool Guides {#tools}

\tableofcontents

# llzk-opt

`llzk-opt` is a version of the [`mlir-opt` tool](https://mlir.llvm.org/docs/Tutorials/MlirOpt/) that supports
passes on LLZK IR files. You can refer to the `mlir-opt` documentation for a general
overview of the operation of `*-opt` tooling, but note that many options and passes
available in `mlir-opt` are not available in `llzk-opt`.
`llzk-opt -h` will show a list of all available flags and options.

## LLZK Pass Documentation

### Analysis Passes

\include{doc,raise=1} build/doc/mlir/passes/AnalysisPasses.md

### Transformation Passes

\include{doc,raise=1} build/doc/mlir/passes/LLZKTransformationPasses.md

### Validation Passes

\include{doc,raise=1} build/doc/mlir/passes/LLZKValidationPasses.md

# llzk-lsp-server

`cmake --build <build dir> --target llzk-lsp-server` will produce an LLZK-specific
LSP server that can be used in an IDE to provide language information for LLZK.
Refer to the [MLIR LSP documentation](https://mlir.llvm.org/docs/Tools/MLIRLSP/) for
a more detailed explanation of the MLIR LSP tools and how to set them up in your IDE.

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
