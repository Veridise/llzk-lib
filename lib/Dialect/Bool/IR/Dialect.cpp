//===-- Dialect.cpp - Boolean dialect implementation ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Bool/IR/Dialect.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Dialect.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Twine.h>

// TableGen'd implementation files
#include "llzk/Dialect/Bool/IR/Dialect.cpp.inc"

// Need a complete declaration of storage classes for below
#define GET_ATTRDEF_CLASSES
#include "llzk/Dialect/Bool/IR/Attrs.cpp.inc"

//===------------------------------------------------------------------===//
// BoolDialect
//===------------------------------------------------------------------===//

auto llzk::boolean::BoolDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Bool/IR/Ops.cpp.inc"
  >();

  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "llzk/Dialect/Bool/IR/Attrs.cpp.inc"
  >();
  // clang-format on
}
