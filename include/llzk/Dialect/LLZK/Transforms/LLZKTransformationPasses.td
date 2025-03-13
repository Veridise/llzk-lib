#ifndef LLZK_TRANSFORMATION_PASSES_TD
#define LLZK_TRANSFORMATION_PASSES_TD

include "mlir/Pass/PassBase.td"

def InlineIncludesPass : Pass<"llzk-inline-includes", "::mlir::ModuleOp"> {
  let summary = "Replace all IncludeOp with contents of the referenced file.";
  let constructor = "llzk::createInlineIncludesPass()";
  let dependentDialects = ["llzk::LLZKDialect"];
}

def RedundantReadAndWriteEliminationPass : Pass<"llzk-duplicate-read-write-elimination", "::mlir::ModuleOp"> {
  let summary = "Remove redundant reads and writes";
  let description = [{
    Remove read and write operations to struct fields and arrays that are redundant or unnecessary.
  }];
  let constructor = "llzk::createRedundantReadAndWriteEliminationPass()";
  let dependentDialects = ["llzk::LLZKDialect"];
}

def RedundantOperationEliminationPass : Pass<"llzk-duplicate-op-elimination", "::mlir::ModuleOp"> {
  let summary = "Remove redundant operations";
  let description = [{
    Remove llzk and arith dialect operations that produce the same results
    as previously executed operations.
    Pass should be used in conjunction with the read and write elimination for
    maximum effect.
  }];
  let constructor = "llzk::createRedundantOperationEliminationPass()";
  let dependentDialects = ["llzk::LLZKDialect"];
}

def UnusedDeclarationEliminationPass : Pass<"llzk-unused-declaration-elimination", "::mlir::ModuleOp"> {
  let summary = "Remove unused field and struct declarations";
  let description = [{
    Remove field and struct declarations that are unused within the current compilation
    unit. Note that this pass may cause linking issues with external modules that
    depend on any unused field and struct declarations from this compilation unit.

    Pass should be used in conjunction with llzk-duplicate-read-write-elimination
    and llzk-duplicate-op-elimination for maximum effect.
  }];
  let constructor = "llzk::createUnusedDeclarationEliminationPass()";
  let dependentDialects = ["llzk::LLZKDialect"];
}

def FlatteningPass : Pass<"llzk-flatten", "::mlir::ModuleOp"> {
  let summary =
      "Replace parameterized structs with flattened (i.e. no parameter) "
      "versions of those structs based on requested return type at calls to "
      "`compute()` functions and unroll loops";
  let constructor = "llzk::createFlatteningPass()";
  let dependentDialects = ["llzk::LLZKDialect"];
}

#endif // LLZK_TRANSFORMATION_PASSES_TD
