#ifndef ZKIR_ZKIRPASSES_TD
#define ZKIR_ZKIRPASSES_TD

include "zkir/Dialect/ZKIR/Util/Constants.td"

include "mlir/Pass/PassBase.td"

def InlineIncludesPass
    : Pass<InlineIncludesPassName.result, "::mlir::ModuleOp"> {
  let summary = InlineIncludesPassSummary.result;
  let constructor = "zkir::createInlineIncludesPass()";
  let dependentDialects = ["zkir::ZKIRDialect"];
}

#endif // ZKIR_ZKIRPASSES_TD
