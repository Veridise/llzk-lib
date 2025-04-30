//===-- Ops.cpp - Constrain operation implementations -----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Function/IR/OpTraits.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Util/BuilderHelper.h"

#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/Twine.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Constrain/IR/Ops.cpp.inc"

using namespace mlir;
using namespace llzk::array;

namespace llzk::constrain {

//===------------------------------------------------------------------===//
// EmitEqualityOp
//===------------------------------------------------------------------===//

LogicalResult EmitEqualityOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(
      tables, *this, ArrayRef<Type> {getLhs().getType(), getRhs().getType()}
  );
}

Type EmitEqualityOp::inferRHS(Type lhsType) { return lhsType; }

//===------------------------------------------------------------------===//
// EmitContainmentOp
//===------------------------------------------------------------------===//

LogicalResult EmitContainmentOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(
      tables, *this, ArrayRef<Type> {getLhs().getType(), getRhs().getType()}
  );
}

Type EmitContainmentOp::inferRHS(Type lhsType) {
  assert(llvm::isa<ArrayType>(lhsType)); // per ODS spec of EmitContainmentOp
  return llvm::cast<ArrayType>(lhsType).getElementType();
}

} // namespace llzk::constrain
