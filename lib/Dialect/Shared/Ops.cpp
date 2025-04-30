//===-- Ops.cpp -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/Shared/Ops.h"

using namespace mlir;

namespace llzk {

using namespace function;

bool isInStruct(Operation *op) { return succeeded(getParentOfType<StructDefOp>(op)); }

FailureOr<StructDefOp> verifyInStruct(Operation *op) {
  FailureOr<StructDefOp> res = getParentOfType<StructDefOp>(op);
  if (failed(res)) {
    return op->emitOpError() << "only valid within a '" << StructDefOp::getOperationName()
                             << "' ancestor";
  }
  return res;
}

bool isInStructFunctionNamed(Operation *op, char const *funcName) {
  FailureOr<FuncDefOp> parentFuncOpt = getParentOfType<FuncDefOp>(op);
  if (succeeded(parentFuncOpt)) {
    FuncDefOp parentFunc = parentFuncOpt.value();
    if (isInStruct(parentFunc.getOperation())) {
      if (parentFunc.getSymName().compare(funcName) == 0) {
        return true;
      }
    }
  }
  return false;
}

// Again, only valid/implemented for StructDefOp
template <> LogicalResult SetFuncAllowAttrs<StructDefOp>::verifyTrait(Operation *structOp) {
  assert(llvm::isa<StructDefOp>(structOp));
  llvm::cast<StructDefOp>(structOp).getBody().walk([](FuncDefOp funcDef) {
    if (funcDef.nameIsConstrain()) {
      funcDef.setAllowConstraintAttr();
      funcDef.setAllowWitnessAttr(false);
    } else if (funcDef.nameIsCompute()) {
      funcDef.setAllowConstraintAttr(false);
      funcDef.setAllowWitnessAttr();
    }
  });
  return success();
}

} // namespace llzk
