#ifndef LLZK_TRANSFORMATION_PASSES_TD
#define LLZK_TRANSFORMATION_PASSES_TD

include "mlir/Pass/PassBase.td"

def DuplicateStructEliminationPass : Pass<"llzk-duplicate-struct-elimination", "::mlir::ModuleOp"> {
  let summary = "Deduplicate struct instantiations";
  let description = [{
    Deduplicate struct instantiations that perform identical computation.
    Redundant calls to @compute will be removed, along with unnecessary struct
    field read operations.
  }];
  let constructor = "llzk::createDuplicateStructEliminationPass()";
  let dependentDialects = ["llzk::LLZKDialect"];
}

def InlineIncludesPass : Pass<"llzk-inline-includes", "::mlir::ModuleOp"> {
  let summary = "Replace all IncludeOp with contents of the referenced file.";
  let constructor = "llzk::createInlineIncludesPass()";
  let dependentDialects = ["llzk::LLZKDialect"];
}

def RedundantReadAndWriteEliminationPass : Pass<"llzk-duplicate-read-write-elimination", "::mlir::ModuleOp"> {
  let summary = "Remove redundant field reads and writes";
  let description = [{
  }];
  let constructor = "llzk::createRedundantReadAndWriteEliminationPass()";
  let dependentDialects = ["llzk::LLZKDialect"];
}

def RedundantFunctionCallEliminationPass : Pass<"llzk-duplicate-func-call-elimination", "::mlir::ModuleOp"> {
  let summary = "Remove redundant function calls";
  let description = [{
  }];
  let constructor = "llzk::createRedundantFunctionCallEliminationPass()";
  let dependentDialects = ["llzk::LLZKDialect"];
}

def RedundantOperationEliminationPass : Pass<"llzk-duplicate-op-elimination", "::mlir::ModuleOp"> {
  let summary = "Remove redundant operations";
  let description = [{
    Pass should be used in conjunction with the read and write elimination.
  }];
  let constructor = "llzk::createRedundantOperationEliminationPass()";
  let dependentDialects = ["llzk::LLZKDialect"];
}

// TODO: add an unnecessary field removal pass
// TODO: add a pipeline for these things

#endif // LLZK_TRANSFORMATION_PASSES_TD
