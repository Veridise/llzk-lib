//===-- Ops.cpp - String operation implementations --------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/String/IR/Ops.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/String/IR/Ops.cpp.inc"

namespace llzk::string {

using namespace mlir;

//===------------------------------------------------------------------===//
// LitStringOp
//===------------------------------------------------------------------===//

OpFoldResult LitStringOp::fold(LitStringOp::FoldAdaptor) { return getValueAttr(); }

} // namespace llzk::string
