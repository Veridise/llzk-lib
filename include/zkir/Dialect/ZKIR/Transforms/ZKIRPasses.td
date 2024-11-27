#ifndef ZKIR_ZKIRPASSES_TD
#define ZKIR_ZKIRPASSES_TD

include "mlir/Pass/PassBase.td"

def InlineIncludesPass : Pass<"zkir-inline-includes", "::mlir::ModuleOp"> {
  let summary = "Replace all IncludeOp with contents of the referenced file.";
  let constructor = "zkir::createInlineIncludesPass()";
  let dependentDialects = ["zkir::ZKIRDialect"];
}

#endif // ZKIR_ZKIRPASSES_TD
