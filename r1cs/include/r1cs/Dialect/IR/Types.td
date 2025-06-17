//===- Types.td ----------------------------------------------------------===//

#ifndef R1CS_TYPES
#define R1CS_TYPES

include "mlir/IR/OpBase.td"
include "r1cs/Dialect/IR/Dialect.td"

class R1CS_Type<string name, string storage> : TypeDef<R1CSDialect, name> {
  let mnemonic = storage;
}

def R1CS_SignalType : R1CS_Type<"Signal", "signal"> {
  let summary = "R1CS signal wire type";
  let parameters = (ins "bool":$isPublic);

  let hasCustomAssemblyFormat = 1;
}

def R1CS_LinearType : R1CS_Type<"Linear", "linear"> {
  let summary = "R1CS linear expression type";
}

#endif // R1CS_TYPES
