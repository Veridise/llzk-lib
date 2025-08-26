//===-- Array.cpp - Array dialect C API implementation ----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/CAPI/Builder.h"
#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Array/Transforms/TransformationPasses.h"

#include "llzk-c/Dialect/Array.h"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>

#include <mlir-c/Pass.h>

using namespace mlir;
using namespace llzk::array;
using namespace llzk;

static void registerLLZKArrayTransformationPasses() { llzk::array::registerTransformationPasses(); }

// Include impl for transformation passes
#include "llzk/Dialect/Array/Transforms/TransformationPasses.capi.cpp.inc"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Array, llzk__array, ArrayDialect)

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

MlirType llzkArrayTypeGet(MlirType elementType, intptr_t nDims, MlirAttribute const *dims) {
  SmallVector<Attribute> dimsSto;
  return wrap(ArrayType::get(unwrap(elementType), unwrapList(nDims, dims, dimsSto)));
}

MlirType
llzkArrayTypeGetWithNumericDims(MlirType elementType, intptr_t nDims, int64_t const *dims) {
  return wrap(ArrayType::get(unwrap(elementType), ArrayRef(dims, nDims)));
}

bool llzkTypeIsAArrayType(MlirType type) { return llvm::isa<ArrayType>(unwrap(type)); }

MlirType llzkArrayTypeGetElementType(MlirType type) {
  return wrap(mlir::unwrap_cast<ArrayType>(type).getElementType());
}

intptr_t llzkArrayTypeGetNumDims(MlirType type) {
  return static_cast<intptr_t>(mlir::unwrap_cast<ArrayType>(type).getDimensionSizes().size());
}

MlirAttribute llzkArrayTypeGetDim(MlirType type, intptr_t idx) {
  return wrap(mlir::unwrap_cast<ArrayType>(type).getDimensionSizes()[idx]);
}

//===----------------------------------------------------------------------===//
// CreateArrayOp
//===----------------------------------------------------------------------===//

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    CreateArrayOp, WithValues, MlirType arrayType, intptr_t nValues, MlirValue const *values
) {
  SmallVector<Value> valueSto;
  return wrap(create<CreateArrayOp>(
      builder, location, mlir::unwrap_cast<ArrayType>(arrayType),
      ValueRange(unwrapList(nValues, values, valueSto))
  ));
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    CreateArrayOp, WithMapOperands, MlirType arrayType, intptr_t nMapOperands,
    MlirValueRange const *mapOperands, MlirAttribute numDimsPerMap
) {
  MapOperandsHelper<> mapOps(nMapOperands, mapOperands);
  return wrap(create<CreateArrayOp>(
      builder, location, mlir::unwrap_cast<ArrayType>(arrayType), *mapOps,
      mlir::unwrap_cast<DenseI32ArrayAttr>(numDimsPerMap)
  ));
}

/// Creates a CreateArrayOp with its size information declared with AffineMaps and operands.
LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    CreateArrayOp, WithMapOperandsAndDims, MlirType arrayType, intptr_t nMapOperands,
    MlirValueRange const *mapOperands, intptr_t nNumsDimsPerMap, int32_t const *numDimsPerMap
) {
  MapOperandsHelper<> mapOps(nMapOperands, mapOperands);
  return wrap(create<CreateArrayOp>(
      builder, location, mlir::unwrap_cast<ArrayType>(arrayType), *mapOps,
      ArrayRef(numDimsPerMap, nNumsDimsPerMap)
  ));
}
