//===-- Global.cpp - Global dialect C API implementation --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llzk/Dialect/Global/IR/Dialect.h>
#include <llzk/Dialect/Global/IR/Ops.h>

#include <mlir/CAPI/Registration.h>

#include <llzk-c/Dialect/Global.h>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Global, llzk__global, llzk::global::GlobalDialect)

//===----------------------------------------------------------------------===//
// GlobalDefOp
//===----------------------------------------------------------------------===//

bool llzkOperationIsAGlobalDefOp(MlirOperation op) {
  return mlir::isa<llzk::global::GlobalDefOp>(unwrap(op));
}

bool llzkGlobalDefOpGetIsConstant(MlirOperation op) {
  return mlir::cast<llzk::global::GlobalDefOp>(unwrap(op)).isConstant();
}
