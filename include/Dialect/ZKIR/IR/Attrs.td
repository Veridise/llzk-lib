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

class ZKIR_EnumAttr<EnumAttrInfo enum, string name, string enumMnemonic,
                    list<Trait> traits = []>
    : EnumAttr<ZKIRDialect, enum, name, traits> {
  let mnemonic = enumMnemonic;
}

def ZKIR_TypeModiferAttr : ZKIR_EnumAttr<ZKIR_TypeModifiers, "modifier", "modifier"> {
  let assemblyFormat = [{ `<` $value `>` }];
}

def ZKIR_CmpPredicateAttr : ZKIR_EnumAttr<ZKIR_FeltCmpPredicate, "cmp_predicate", "cmp"> {
  let assemblyFormat = [{ $value }];
}

def ZKIR_FeltConstAttr : ZKIR_Attr<"FeltConst", "feltconst", [TypedAttrInterface]> {
  let summary = "finite field element";
  let description = [{
    A felt attribute represents a finite field element.
  }];

  let parameters = (ins "::llvm::APInt":$value);

  let assemblyFormat = [{ custom<APInt>($value) }];

  // let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    ::mlir::Type getType() const;
  }];
}

#endif // ZKIR_ATTRS
