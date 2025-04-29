//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_STRING_OPS
#define LLZK_STRING_OPS

include "llzk/Dialect/String/IR/Dialect.td"
include "llzk/Dialect/String/IR/Types.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def LLZK_LitStringOp : Op<StringDialect, "new", [Pure, ConstantLike]> {
  let summary = "literal string";
  let arguments = (ins StrAttr:$value);
  let results = (outs LLZK_String:$result);

  let assemblyFormat = [{
    $value attr-dict
  }];
  let hasFolder = 1;
}

#endif // LLZK_STRING_OPS
