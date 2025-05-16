//===-- Felt.cpp - Felt dialect C API implementation ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llzk/Dialect/Felt/IR/Dialect.h>
#include <llzk/Dialect/Felt/IR/Types.h>

#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>

using namespace llzk::felt;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Felt, llzk__felt, FeltDialect)

MlirType llzkFeltTypeGet(MlirContext ctx) { return wrap(FeltType::get(unwrap(ctx))); }
