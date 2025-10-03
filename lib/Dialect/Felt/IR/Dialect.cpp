//===-- Dialect.cpp - Felt dialect implementation ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/LLZK/IR/Versioning.h"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

// TableGen'd implementation files
#include "llzk/Dialect/Felt/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/Felt/IR/Types.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "llzk/Dialect/Felt/IR/Attrs.cpp.inc"

//===------------------------------------------------------------------===//
// FeltDialect
//===------------------------------------------------------------------===//

auto llzk::felt::FeltDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Felt/IR/Ops.cpp.inc"
  >();

  addTypes<
    #define GET_TYPEDEF_LIST
    #include "llzk/Dialect/Felt/IR/Types.cpp.inc"
  >();

  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "llzk/Dialect/Felt/IR/Attrs.cpp.inc"
  >();
  // clang-format on
  addInterfaces<LLZKDialectBytecodeInterface<FeltDialect>>();
}

//===------------------------------------------------------------------===//
// FeltConstAttr
//===------------------------------------------------------------------===//

llzk::felt::FeltConstAttr llzk::felt::FeltConstAttr::get(
    ::mlir::MLIRContext *context, unsigned numBits, llvm::StringRef str
) {
  return Base::get(context, llvm::APInt(numBits, str, 10));
}

llzk::felt::FeltConstAttr llzk::felt::FeltConstAttr::get(
    mlir::MLIRContext *context, unsigned numBits, llvm::ArrayRef<uint64_t> parts
) {
  return Base::get(context, llvm::APInt(numBits, parts));
}
