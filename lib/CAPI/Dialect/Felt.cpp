//===-- Felt.cpp - Felt dialect C API implementation ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llzk/Dialect/Felt/IR/Attrs.h>
#include <llzk/Dialect/Felt/IR/Dialect.h>
#include <llzk/Dialect/Felt/IR/Types.h>
#include <llzk/Dialect/LLZK/IR/AttributeHelper.h>

#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>

#include <llzk-c/Dialect/Felt.h>

using namespace llzk::felt;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Felt, llzk__felt, FeltDialect)

MlirAttribute llzkFeltConstAttrGet(MlirContext ctx, int64_t value) {
  return wrap(FeltConstAttr::get(unwrap(ctx), llzk::toAPInt(value)));
}

bool llzkAttributeIsAFeltConstAttr(MlirAttribute attr) {
  return mlir::isa<FeltConstAttr>(unwrap(attr));
}

MlirType llzkFeltTypeGet(MlirContext ctx) { return wrap(FeltType::get(unwrap(ctx))); }

bool llzkTypeIsAFeltType(MlirType type) { return mlir::isa<FeltType>(unwrap(type)); }
