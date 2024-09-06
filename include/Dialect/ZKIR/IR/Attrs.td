#ifndef ZKIR_ATTRS
#define ZKIR_ATTRS

include "Dialect/ZKIR/IR/Dialect.td"
include "Dialect/ZKIR/IR/Enums.td"
include "Dialect/ZKIR/IR/Types.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"

class ZKIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ZKIRDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif
