//===-- Ops.cpp - Felt operation implementations ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Types.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/SmallString.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Felt/IR/Ops.cpp.inc"

using namespace mlir;

namespace llzk::felt {

//===------------------------------------------------------------------===//
// FeltConstantOp
//===------------------------------------------------------------------===//

void FeltConstantOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  llvm::SmallString<32> buf;
  llvm::raw_svector_ostream os(buf);
  os << "felt_const_";
  getValue().getValue().toStringUnsigned(buf);
  setNameFn(getResult(), buf);
}

OpFoldResult FeltConstantOp::fold(FeltConstantOp::FoldAdaptor) { return getValue(); }

//===------------------------------------------------------------------===//
// FeltNonDetOp
//===------------------------------------------------------------------===//

void FeltNonDetOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "felt_nondet");
}

} // namespace llzk::felt
