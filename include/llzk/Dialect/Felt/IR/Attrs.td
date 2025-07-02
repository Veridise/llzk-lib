//===-- Attrs.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_FELT_ATTRS
#define LLZK_FELT_ATTRS

include "llzk/Dialect/Felt/IR/Dialect.td"
include "llzk/Dialect/LLZK/IR/AttributeHelper.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def LLZK_FeltConstAttr
    : AttrDef<FeltDialect, "FeltConst", [TypedAttrInterface]> {
  let mnemonic = "const";
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

#endif // LLZK_FELT_ATTRS
