//===-- Dialect.cpp - Include dialect implementation ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Include/IR/Dialect.h"
#include "llzk/Dialect/Include/IR/Ops.h"
#include "llzk/Util/BuilderHelper.h"

#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Twine.h>

// TableGen'd implementation files
#include "llzk/Dialect/Include/IR/Dialect.cpp.inc"

//===------------------------------------------------------------------===//
// IncludeDialect
//===------------------------------------------------------------------===//

auto llzk::include::IncludeDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Include/IR/Ops.cpp.inc"
  >();
  // clang-format on
}
