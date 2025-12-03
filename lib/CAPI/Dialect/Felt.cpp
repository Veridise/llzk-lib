//===-- Felt.cpp - Felt dialect C API implementation ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"

#include "llzk-c/Dialect/Felt.h"

#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::felt;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Felt, llzk__felt, FeltDialect)

MlirAttribute llzkFeltConstAttrGet(MlirContext ctx, int64_t value) {
  return wrap(FeltConstAttr::get(unwrap(ctx), toAPInt(value)));
}

MlirAttribute llzkFeltConstAttrGetWithBits(MlirContext ctx, unsigned numBits, int64_t value) {
  return wrap(FeltConstAttr::get(unwrap(ctx), APInt(numBits, value)));
}

MlirAttribute llzkFeltConstAttrGetFromString(MlirContext ctx, unsigned numBits, MlirStringRef str) {
  return wrap(FeltConstAttr::get(unwrap(ctx), numBits, unwrap(str)));
}

MlirAttribute llzkFeltConstAttrGetFromParts(
    MlirContext context, unsigned numBits, const uint64_t *parts, intptr_t nParts
) {
  return wrap(FeltConstAttr::get(unwrap(context), numBits, ArrayRef(parts, nParts)));
}

bool llzkAttributeIsAFeltConstAttr(MlirAttribute attr) {
  return llvm::isa<FeltConstAttr>(unwrap(attr));
}

MlirType llzkFeltTypeGet(MlirContext ctx) { return wrap(FeltType::get(unwrap(ctx))); }

bool llzkTypeIsAFeltType(MlirType type) { return llvm::isa<FeltType>(unwrap(type)); }
