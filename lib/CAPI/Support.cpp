//===-- Support.cpp - C API general utilities ---------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/CAPI/Support.h"
#include "llzk/Util/SymbolLookup.h"

#include "llzk-c/Support.h"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/CAPI/Wrap.h>
#include <mlir/IR/Iterators.h>

using namespace llzk;
using namespace mlir;

/// Destroys the lookup result, releasing its resources.
void llzkSymbolLookupResultDestroy(LlzkSymbolLookupResult result) {
  delete reinterpret_cast<SymbolLookupResultUntyped *>(result.ptr);
}

/// Returns the looked up Operation.
///
/// The lifetime of the Operation is tied to the lifetime of the lookup result.
MlirOperation LlzkSymbolLookupResultGetOperation(LlzkSymbolLookupResult wrapped) {
  SymbolLookupResultUntyped *result = reinterpret_cast<SymbolLookupResultUntyped *>(wrapped.ptr);
  return wrap(result->get());
}

/// Note: Duplicated from upstream LLVM. Available in 21.1.8 and later.
void mlirOperationReplaceUsesOfWith(MlirOperation op, MlirValue oldValue, MlirValue newValue) {
  unwrap(op)->replaceUsesOfWith(unwrap(oldValue), unwrap(newValue));
}

/// Note: Duplicated from upstream LLVM.
static mlir::WalkResult unwrap(MlirWalkResult result) {
  switch (result) {
  case MlirWalkResultAdvance:
    return mlir::WalkResult::advance();

  case MlirWalkResultInterrupt:
    return mlir::WalkResult::interrupt();

  case MlirWalkResultSkip:
    return mlir::WalkResult::skip();
  }
  llvm_unreachable("unknown result in WalkResult::unwrap");
}

void mlirOperationWalkReverse(
    MlirOperation from, MlirOperationWalkCallback callback, void *userData, MlirWalkOrder walkOrder
) {
  switch (walkOrder) {
  case MlirWalkPreOrder:
    unwrap(from)->walk<WalkOrder::PreOrder, ReverseIterator>([callback, userData](Operation *op) {
      return unwrap(callback(wrap(op), userData));
    });
    break;
  case MlirWalkPostOrder:
    unwrap(from)->walk<WalkOrder::PostOrder, ReverseIterator>([callback, userData](Operation *op) {
      return unwrap(callback(wrap(op), userData));
    });
  }
}
