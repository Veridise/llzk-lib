//===-- Attrs.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_ATTRS
#define LLZK_ATTRS

include "llzk/Dialect/LLZK/IR/Dialect.td"
include "llzk/Dialect/LLZK/IR/Enums.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"

// APInts that allows comparison involving different bitwidths.
class APIntParameter<string desc = "">
    : AttrOrTypeParameter<"::llvm::APInt", desc> {
  let comparator = "::llvm::APInt::isSameValue($_lhs, $_rhs)";
}

class LLZK_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<LLZKDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class LLZK_EnumAttr<EnumAttrInfo enum, string name, string enumMnemonic,
                    list<Trait> traits = []>
    : EnumAttr<LLZKDialect, enum, name, traits> {
  let mnemonic = enumMnemonic;
}

def LLZK_CmpPredicateAttr
    : LLZK_EnumAttr<LLZK_FeltCmpPredicate, "cmp_predicate", "cmp"> {
  let assemblyFormat = [{ $value }];
}

def LLZK_FeltConstAttr
    : LLZK_Attr<"FeltConst", "constfelt", [TypedAttrInterface]> {
  let summary = "finite field element";
  let description = [{
    A felt attribute represents a finite field element.
  }];

  let parameters = (ins APIntParameter<"The felt constant value">:$value);

  let assemblyFormat = [{ $value }];

  let extraClassDeclaration = [{
    ::mlir::Type getType() const;
  }];
}

def LLZK_PubModifierAttr : LLZK_Attr<"Public", "pub"> {
  let summary = "A unit attribute to mark a type as public";
  let description = [{
    Examples:

    ```llzk
    llzk.field @field_name : !llzk.felt {llzk.pub}

    llzk.func @func_name(%0: !llzk.felt {llzk.pub})
    ```
  }];
}

def LLZK_LoopBounds : LLZK_Attr<"LoopBounds", "loopbounds"> {
  let summary = "Annotation with the bounds of a loop";
  let description = [{
    This attribute holds information useful for the analysis of loops.
    Holds the bounds of the loop and the step size.

    Example:

    ```llzk
    scf.while ... {
      ...
    } do {
      ...
    } attributes { llzk.loopbounds = #llzk.loopbounds<0 to 10 step 1> }
    ```
  }];

  let parameters =
      (ins APIntParameter<"Loop variable lower bound (inclusive)">:$lower,
          APIntParameter<"Loop variable upper bound (exclusive)">:$upper,
          APIntParameter<"Loop variable step/increment">:$step);

  let assemblyFormat = "`<` $lower `to` $upper `step` $step `>`";
}

#endif // LLZK_ATTRS
