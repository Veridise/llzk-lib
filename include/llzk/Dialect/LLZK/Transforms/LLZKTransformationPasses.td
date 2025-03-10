#ifndef LLZK_TRANSFORMATION_PASSES_TD
#define LLZK_TRANSFORMATION_PASSES_TD

include "mlir/Pass/PassBase.td"

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

def RedundantOperationEliminationPass : Pass<"llzk-duplicate-op-elimination", "::mlir::ModuleOp"> {
  let summary = "Remove redundant operations";
  let description = [{
    Pass should be used in conjunction with the read and write elimination.
  }];
  let constructor = "llzk::createRedundantOperationEliminationPass()";
  let dependentDialects = ["llzk::LLZKDialect"];
}

#endif // LLZK_TRANSFORMATION_PASSES_TD
