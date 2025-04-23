//===-- ErrorHelper.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/ErrorHandling.h>

namespace llzk {

using EmitErrorFn = llvm::function_ref<mlir::InFlightDiagnostic()>;

// This type is required by the functions below to take ownership of the lambda so it is not
// destroyed upon return from the function. It can be implicitly converted to EmitErrorFn.
using OwningEmitErrorFn = std::function<mlir::InFlightDiagnostic()>;

inline OwningEmitErrorFn getEmitOpErrFn(mlir::Operation *op) {
  return [op]() { return op->emitOpError(); };
}

template <typename TypeClass> inline OwningEmitErrorFn getEmitOpErrFn(TypeClass *opImpl) {
  return getEmitOpErrFn(opImpl->getOperation());
}

inline void ensure(bool condition, llvm::Twine errMsg) {
  if (!condition) {
    llvm::report_fatal_error(errMsg);
  }
}

} // namespace llzk
