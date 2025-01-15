#ifndef LLZK_VALIDATION_PASSES_TD
#define LLZK_VALIDATION_PASSES_TD

include "mlir/Pass/PassBase.td"

def FieldWriteValidatorPass
    : Pass<"llzk-validate-field-writes", "::llzk::StructDefOp"> {
  let summary =
      "Detect multiple and missing writes to the same field of a component.";
  let constructor = "llzk::createFieldWriteValidatorPass()";
  let dependentDialects = ["llzk::LLZKDialect"];
}

#endif // LLZK_VALIDATION_PASSES_TD
