//===-- Array.h - C API for Array dialect -------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Array dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_ARRAY_H
#define LLZK_C_DIALECT_ARRAY_H

#include "llzk/Dialect/Array/Transforms/TransformationPasses.capi.h.inc"

#include <llzk-c/Support.h>
#include <mlir-c/IR.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Array, llzk__array);

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

/// Creates a llzk::array::ArrayType using a list of attributes as dimensions
MLIR_CAPI_EXPORTED MlirType llzkArrayTypeGet(MlirType, intptr_t, MlirAttribute const *);

/// Returns true of the type is a llzk::array::ArrayType.
LLZK_DECLARE_TYPE_ISA(ArrayType);

/// Creates a llzk::array::ArrayType using a list of numbers as dimensions
MLIR_CAPI_EXPORTED MlirType llzkArrayTypeGetWithNumericDims(MlirType, intptr_t, int64_t const *);

/// Returns the element type of a llzk::array::ArrayType.
MLIR_CAPI_EXPORTED MlirType llzkArrayTypeGetElementType(MlirType);

/// Returns the number of dimensions of a llzk::array::ArrayType.
MLIR_CAPI_EXPORTED intptr_t llzkArrayTypeGetNumDims(MlirType);

/// Returns the n-th dimention of a llzk::array::ArrayType.
MLIR_CAPI_EXPORTED MlirAttribute llzkArrayTypeGetDim(MlirType, intptr_t);

//===----------------------------------------------------------------------===//
// CreateArrayOp
//===----------------------------------------------------------------------===//

/// Creates a CreateArrayOp from a list of Values.
LLZK_DECLARE_OP_BUILD_METHOD(CreateArrayOp, WithValues, MlirType, intptr_t, MlirValue const *);

/// Creates a CreateArrayOp with its size information declared with AffineMaps and operands.
/// The Attribute argument must be a DenseI32ArrayAttr.
LLZK_DECLARE_OP_BUILD_METHOD(
    CreateArrayOp, WithMapOperands, MlirType, intptr_t, MlirValueRange const *, MlirAttribute
);

/// Creates a CreateArrayOp with its size information declared with AffineMaps and operands.
LLZK_DECLARE_OP_BUILD_METHOD(
    CreateArrayOp, WithMapOperandsAndDims, MlirType, intptr_t, MlirValueRange const *, intptr_t,
    int32_t const *
);

#ifdef __cplusplus
}
#endif

#endif
