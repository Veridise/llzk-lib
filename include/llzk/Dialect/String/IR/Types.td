//===-- Types.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_STRING_TYPES
#define LLZK_STRING_TYPES

include "llzk/Dialect/String/IR/Dialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def LLZK_String : TypeDef<StringDialect, "String", []> {
  let mnemonic = "string";
  let summary = "string type";
}

#endif // LLZK_STRING_TYPES
