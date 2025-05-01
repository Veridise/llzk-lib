//===-- Enums.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_BOOLEAN_ENUMS
#define LLZK_BOOLEAN_ENUMS

include "mlir/IR/EnumAttr.td"

class LLZK_I32Enum<string name, string desc, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, desc, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::llzk::boolean";
}

def LLZK_FeltCmpPredicate
    : LLZK_I32Enum<
          "FeltCmpPredicate", "Field element comparison predicate",
          [I32EnumAttrCase<"EQ", 0, "eq">, I32EnumAttrCase<"NE", 1, "ne">,
           I32EnumAttrCase<"LT", 2, "lt">, I32EnumAttrCase<"LE", 3, "le">,
           I32EnumAttrCase<"GT", 4, "gt">, I32EnumAttrCase<"GE", 5, "ge">,
]>;

#endif // LLZK_BOOLEAN_ENUMS
