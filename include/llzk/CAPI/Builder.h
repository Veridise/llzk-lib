//===-- Builder.h - C API for op builder ------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk-c/Builder.h"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Wrap.h>
#include <mlir/IR/Builders.h>

DEFINE_C_API_PTR_METHODS(MlirOpBuilder, mlir::OpBuilder)

namespace llzk {

// Taken from mlir/IR/Builders.h
template <typename OpT>
mlir::RegisteredOperationName getCheckRegisteredInfo(mlir::MLIRContext *ctx) {
  std::optional<mlir::RegisteredOperationName> opName =
      mlir::RegisteredOperationName::lookup(OpT::getOperationName(), ctx);
  if (LLVM_UNLIKELY(!opName)) {
    llvm::report_fatal_error(
        "Building op `" + OpT::getOperationName() +
        "` but it isn't known in this MLIRContext: the dialect may not "
        "be loaded or this operation hasn't been added by the dialect. See "
        "also https://mlir.llvm.org/getting_started/Faq/"
        "#registered-loaded-dependent-whats-up-with-dialects-management"
    );
  }
  return *opName;
}

/// Creates a new operation using an ODS build method.
template <typename OpTy, typename... Args>
mlir::Operation *create(MlirOpBuilder cBuilder, MlirLocation cLocation, Args &&...args) {
  auto location = unwrap(cLocation);
  auto *builder = unwrap(cBuilder);
  assert(builder);
  mlir::OperationState state(location, getCheckRegisteredInfo<OpTy>(location.getContext()));
  OpTy::build(*builder, state, std::forward<Args>(args)...);
  auto *op = mlir::Operation::create(state);
  assert(llvm::isa<OpTy>(op) && "builder didn't return the right type");
  return op;
}

} // namespace llzk
