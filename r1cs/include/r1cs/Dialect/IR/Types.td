//===- Types.td ----------------------------------------------------------===//

#ifndef R1CS_TYPES
#define R1CS_TYPES

include "mlir/IR/OpBase.td"
include "r1cs/Dialect/IR/Dialect.td"

def R1CS_SignalType : TypeDef<R1CSDialect, "Signal", []> {
  let summary = "A signal variable in the R1CS system";
  let mnemonic = "signal";
}

def R1CS_LinearType : TypeDef<R1CSDialect, "Linear", []> {
  let summary = "A linear combination of signals";
  let mnemonic = "linear";
}

#endif // R1CS_TYPES