//===-- Support.cpp - C API general utilities ---------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Util/SymbolLookup.h"
#include "llzk/CAPI/Support.h"

#include "llzk-c/Support.h"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/CAPI/Wrap.h>

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

/// Note: Duplicated from upstream LLVM. Available in 21.1.8 and later.
void mlirOperationReplaceUsesOfWith(MlirOperation op, MlirValue oldValue, MlirValue newValue) {
  unwrap(op)->replaceUsesOfWith(unwrap(oldValue), unwrap(newValue));
}
