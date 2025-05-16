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

#include <stdint.h>

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Array, llzk__array);

/// Creates a llzk::array::ArrayType using a list of attributes as dimensions
MLIR_CAPI_EXPORTED MlirType llzkArrayTypeGet(MlirType, intptr_t, MlirAttribute const *);

/// Returns true of the type is a llzk::array::ArrayType.
MLIR_CAPI_EXPORTED bool llzkTypeIsALlzkArrayType(MlirType);

/// Creates a llzk::array::ArrayType using a list of numbers as dimensions
MLIR_CAPI_EXPORTED MlirType llzkArrayTypeGetWithNumericDis(MlirType, intptr_t, int64_t const *);

/// Returns the element type of a llzk::array::ArrayType.
MLIR_CAPI_EXPORTED MlirType llzkArrayTypeGetElementType(MlirType);

/// Returns the number of dimensions of a llzk::array::ArrayType.
MLIR_CAPI_EXPORTED intptr_t llzkArrayTypeGetNumDims(MlirType);

/// Returns the n-th dimention of a llzk::array::ArrayType.
MLIR_CAPI_EXPORTED MlirAttribute llzkArrayTypeGetDim(MlirType, intptr_t);

#ifdef __cplusplus
}
#endif

#endif
