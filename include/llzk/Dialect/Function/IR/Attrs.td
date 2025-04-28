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

include "llzk/Dialect/Function/IR/Dialect.td"

include "mlir/IR/AttrTypeBase.td"

class LLZK_Function_Attr<string name, string attrMnemonic,
                         list<Trait> traits = []>
    : AttrDef<FunctionDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def LLZK_AllowConstraintsAttr
    : LLZK_Function_Attr<"AllowConstraints", "allow_constraints"> {
  let summary = "Marks functions that can emit constraints.";
  let description = [{
    A unit attribute for a function definition to indicate that emit constraint ops are allowed within its body.
  }];
}

#endif // LLZK_ATTRS
