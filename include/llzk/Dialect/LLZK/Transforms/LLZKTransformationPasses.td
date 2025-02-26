#ifndef LLZK_TRANSFORMATION_PASSES_TD
#define LLZK_TRANSFORMATION_PASSES_TD

include "mlir/Pass/PassBase.td"

def InlineIncludesPass : Pass<"llzk-inline-includes", "::mlir::ModuleOp"> {
  let summary = "Replace all IncludeOp with contents of the referenced file.";
  let constructor = "llzk::createInlineIncludesPass()";
  let dependentDialects = ["llzk::LLZKDialect"];
}

def InstantiateStructsPass : Pass<"llzk-flatten", "::mlir::ModuleOp"> {
  let summary =
      "Replace parameterized structs with flattened (i.e. no parameter) "
      "versions of those structs based on requested return type at calls to "
      "`compute()` functions and unroll loops";
  let constructor = "llzk::createFlatteningPass()";
  let dependentDialects = ["llzk::LLZKDialect"];
}

#endif // LLZK_TRANSFORMATION_PASSES_TD
