#ifndef ZKIR_OPS
#define ZKIR_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "Dialect/ZKIR/IR/Dialect.td"
include "Dialect/ZKIR/IR/Attrs.td"
include "Dialect/ZKIR/IR/Types.td"

class ZKIR_Op<string mnemonic, list<Trait> traits = []>
    : Op<ZKIRDialect, mnemonic, traits>;

class ZKIR_BinaryOp<string mnemonic, list<Trait> traits = []>
    : ZKIR_Op<mnemonic,
              !listconcat(traits, [SameOperandsAndResultType, Pure])> {}

#endif // ZKIR_OPS
