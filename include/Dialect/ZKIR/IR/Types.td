#ifndef ZKIR_TYPES
#define ZKIR_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"
include "Dialect/ZKIR/IR/Attrs.td"
include "Dialect/ZKIR/IR/Dialect.td"
include "Dialect/ZKIR/IR/Enums.td"

class ZKIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ZKIRDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif
