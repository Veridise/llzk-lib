//===-- Attrs.cpp - Felt Attr method implementations ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Types.h"

namespace llzk::felt {

mlir::Type FeltConstAttr::getType() const { return FeltType::get(this->getContext()); }

FeltConstAttr
FeltConstAttr::get(::mlir::MLIRContext *context, unsigned numBits, ::llvm::StringRef str) {
  return Base::get(context, llvm::APInt(numBits, str, 10));
}

FeltConstAttr FeltConstAttr::get(
    ::mlir::MLIRContext *context, unsigned numBits, ::llvm::ArrayRef<uint64_t> parts
) {
  return Base::get(context, llvm::APInt(numBits, parts));
}

} // namespace llzk::felt
