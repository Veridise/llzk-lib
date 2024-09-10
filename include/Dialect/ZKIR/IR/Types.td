#ifndef ZKIR_TYPES
#define ZKIR_TYPES

include "Dialect/ZKIR/IR/Attrs.td"
include "Dialect/ZKIR/IR/Dialect.td"
include "Dialect/ZKIR/IR/Enums.td"

class ZKIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ZKIRDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def ZKIR_FeltType : ZKIR_Type<"Felt", "felt"> {
  let summary = "finite field element";

  let extraClassDeclaration = [{
    static ::zkir::FeltType get(::mlir::MLIRContext *context) {
      return Base::get(context);
    }
  }];
}

#endif // ZKIR_TYPES
