#ifndef LLZK_LLZKPASSES_TD
#define LLZK_LLZKPASSES_TD

include "mlir/Pass/PassBase.td"

def InlineIncludesPass : Pass<"llzk-inline-includes", "::mlir::ModuleOp"> {
  let summary = "Replace all IncludeOp with contents of the referenced file.";
  let constructor = "llzk::createInlineIncludesPass()";
  let dependentDialects = ["llzk::LLZKDialect"];
}

#endif // LLZK_LLZKPASSES_TD
