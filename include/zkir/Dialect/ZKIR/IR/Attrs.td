#ifndef ZKIR_ATTRS
#define ZKIR_ATTRS

include "zkir/Dialect/ZKIR/IR/Dialect.td"
include "zkir/Dialect/ZKIR/IR/Enums.td"
include "zkir/Dialect/ZKIR/IR/Types.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"

// APInts that allows comparison involving different bitwidths.
class APIntParameter<string desc> : AttrOrTypeParameter<"::llvm::APInt", desc> {
  let comparator = "::llvm::APInt::isSameValue($_lhs, $_rhs)";
}

class ZKIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ZKIRDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class ZKIR_EnumAttr<EnumAttrInfo enum, string name, string enumMnemonic,
                    list<Trait> traits = []>
    : EnumAttr<ZKIRDialect, enum, name, traits> {
  let mnemonic = enumMnemonic;
}

def ZKIR_CmpPredicateAttr
    : ZKIR_EnumAttr<ZKIR_FeltCmpPredicate, "cmp_predicate", "cmp"> {
  let assemblyFormat = [{ $value }];
}

def ZKIR_FeltConstAttr
    : ZKIR_Attr<"FeltConst", "constfelt", [TypedAttrInterface]> {
  let summary = "finite field element";
  let description = [{
    A felt attribute represents a finite field element.
  }];

  let parameters = (ins APIntParameter<"">:$value);

  let assemblyFormat = [{ custom<APInt>($value) }];

  let extraClassDeclaration = [{
    ::mlir::Type getType() const;
  }];
}

def ZKIR_PubModifierAttr : ZKIR_Attr<"Public", "pub"> {
  let summary = "A unit attribute to mark a type as public";
  let description = [{
    Examples:

    ```zkir
    zkir.field @field_name : !zkir.felt {zkir.pub}

    zkir.func @func_name(%0: !zkir.felt {zkir.pub})
    ```
  }];
}

def ZKIR_LoopBounds : ZKIR_Attr<"LoopBounds", "loopbounds"> {
  let summary = "Annotation with the bounds of a loop";
  let description = [{
    This attribute holds information useful for the analysis of loops.
    Holds the bounds of the loop and the step size.

    Example:

    ```zkir
    scf.while ... {
      ...
    } do {
      ...
    } attributes { zkir.loopbounds = #zkir.loopbounds<0 to 10 step 1> }
    ```
  }];

  let parameters = (ins APIntParameter<"">:$lower, APIntParameter<"">:$upper,
      APIntParameter<"">:$step);

  let assemblyFormat = "`<` custom<APInt>($lower) `to` custom<APInt>($upper) "
                       "`step` custom<APInt>($step) `>`";
}

#endif // ZKIR_ATTRS
