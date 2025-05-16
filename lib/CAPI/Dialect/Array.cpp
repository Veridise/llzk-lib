//===-- Array.cpp - Array dialect C API implementation ----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llzk/Dialect/Array/IR/Types.h>
#include <llzk/Dialect/Array/Transforms/TransformationPasses.h>

#include <mlir/CAPI/Pass.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>

#include <llzk-c/Dialect/Array.h>
#include <mlir-c/Pass.h>

using namespace mlir;
using namespace llzk::array;

// Include impl for transformation passes
#include <llzk/Dialect/Array/Transforms/TransformationPasses.capi.cpp.inc>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Array, llzk__array, ArrayDialect)

MlirType llzkArrayTypeGet(MlirType elementType, intptr_t nDims, MlirAttribute const *dims) {
  SmallVector<Attribute> dimsSto;
  return wrap(ArrayType::get(unwrap(elementType), unwrapList(nDims, dims, dimsSto)));
}

MlirType llzkArrayTypeGetWithNumericDis(MlirType elementType, intptr_t nDims, int64_t const *dims) {
  return wrap(ArrayType::get(unwrap(elementType), ArrayRef(dims, nDims)));
}

bool llzkTypeIsALlzkArrayType(MlirType type) { return mlir::isa<ArrayType>(unwrap(type)); }

MlirType llzkArrayTypeGetElementType(MlirType type) {
  return wrap(mlir::cast<ArrayType>(unwrap(type)).getElementType());
}

intptr_t llzkArrayTypeGetNumDims(MlirType type) {
  return mlir::cast<ArrayType>(unwrap(type)).getDimensionSizes().size();
}

MlirAttribute llzkArrayTypeGetDim(MlirType type, intptr_t idx) {
  return wrap(mlir::cast<ArrayType>(unwrap(type)).getDimensionSizes()[idx]);
}
