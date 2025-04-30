//===-- Ops.cpp - Cast operation implementations ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Cast/IR/Ops.h"
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
#include "llzk/Dialect/Cast/IR/Ops.cpp.inc"

using namespace mlir;

namespace llzk::cast {

//===------------------------------------------------------------------===//
// FeltToIndexOp
//===------------------------------------------------------------------===//

// TODO-IAN: Remove this once the check is added to the FuncDefOp region verifier.
// LogicalResult FeltToIndexOp::verify() {
//   if (auto parentOr = getParentOfType<FuncDefOp>(*this);
//       succeeded(parentOr) && parentOr->isStructConstrain()) {
//     // Traverse the def-use chain to see if this operand, which is a felt, ever
//     // derives from a Signal struct.
//     SmallVector<Value, 2> frontier {getValue()};
//     DenseSet<Value> visited;

//     while (!frontier.empty()) {
//       Value v = frontier.pop_back_val();
//       if (visited.contains(v)) {
//         continue;
//       }
//       visited.insert(v);

//       if (Operation *op = v.getDefiningOp()) {
//         if (FieldReadOp readf = mlir::dyn_cast<FieldReadOp>(op);
//             readf && isSignalType(readf.getComponent().getType())) {
//           return emitOpError()
//               .append("input is derived from a Signal struct, which is illegal in struct
//               constrain "
//                       "function")
//               .attachNote(readf.getLoc())
//               .append("Signal struct value is read here");
//         }
//         frontier.insert(frontier.end(), op->operand_begin(), op->operand_end());
//       }
//     }
//   }

//   return success();
// }

} // namespace llzk::cast
