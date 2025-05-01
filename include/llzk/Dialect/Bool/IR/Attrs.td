//===-- Attrs.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_BOOLEAN_ATTRS
#define LLZK_BOOLEAN_ATTRS

include "llzk/Dialect/Bool/IR/Dialect.td"
include "llzk/Dialect/Bool/IR/Enums.td"

include "mlir/IR/AttrTypeBase.td"

class LLZK_EnumAttr<EnumAttrInfo enum, string name, string enumMnemonic,
                    list<Trait> traits = []>
    : EnumAttr<BoolDialect, enum, name, traits> {
  let mnemonic = enumMnemonic;
}

def LLZK_CmpPredicateAttr
    : LLZK_EnumAttr<LLZK_FeltCmpPredicate, "cmp_predicate", "cmp"> {
  let assemblyFormat = [{ $value }];
}

#endif // LLZK_BOOLEAN_ATTRS
