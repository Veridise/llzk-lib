//===-- Enums.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_ENUMS
#define LLZK_ENUMS

include "mlir/IR/EnumAttr.td"

/// Multiple selection attributes, i.e. bit flags
class LLZK_I32BitEnum<string name, string desc, list<BitEnumAttrCaseBase> cases>
    : I32BitEnumAttr<name, desc, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "llzk";
}

#endif // LLZK_ENUMS
