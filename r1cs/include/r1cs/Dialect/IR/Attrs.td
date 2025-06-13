
#ifndef R1CS_ATTRS
#define R1CS_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "r1cs/Dialect/IR/Dialect.td"

def FeltAttr : AttrDef<R1CSDialect, "Felt"> {
  let summary = "Field element represented as an arbitrary precision integer";
  let cppClassName = "FeltAttr";
  let mnemonic = "felt"; // This is required when using assemblyFormat
  let parameters = (ins "mlir::IntegerAttr":$value);

  let assemblyFormat = "$value";
}

#endif // R1CS_ATTRS
