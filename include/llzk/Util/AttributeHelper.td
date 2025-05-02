//===-- AttributeHelper.td ---------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_ATTRIBUTE_HELPER
#define LLZK_ATTRIBUTE_HELPER

include "mlir/IR/AttrTypeBase.td"

// APInts that allows comparison involving different bitwidths.
class APIntParameter<string desc = "">
    : AttrOrTypeParameter<"::llvm::APInt", desc> {
  let comparator = "::llvm::APInt::isSameValue($_lhs, $_rhs)";
}

#endif // LLZK_ATTRIBUTE_HELPER
