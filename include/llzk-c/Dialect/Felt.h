//===-- Felt.h - C API for Felt dialect ---------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Felt dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations, or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_FELT_H
#define LLZK_C_DIALECT_FELT_H

#include "llzk-c/Support.h"

#include <mlir-c/IR.h>

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Felt, llzk__felt);

/// Creates a llzk::felt::FeltConstAttr.
MLIR_CAPI_EXPORTED MlirAttribute llzkFeltConstAttrGet(MlirContext context, int64_t value);

/// Creates a llzk::felt::FeltConstAttr with a set bit length.
MLIR_CAPI_EXPORTED MlirAttribute
llzkFeltConstAttrGetWithBits(MlirContext ctx, unsigned numBits, int64_t value);

/// Creates a llzk::felt::FeltConstAttr from a base-10 representation of a number.
MLIR_CAPI_EXPORTED MlirAttribute
llzkFeltConstAttrGetFromString(MlirContext context, unsigned numBits, MlirStringRef str);

/// Creates a llzk::felt::FeltConstAttr from an array of big-integer parts in LSB order.
MLIR_CAPI_EXPORTED MlirAttribute llzkFeltConstAttrGetFromParts(
    MlirContext context, unsigned numBits, const uint64_t *parts, intptr_t nParts
);

/// Returns true if the attribute is a FeltConstAttr.
LLZK_DECLARE_ATTR_ISA(FeltConstAttr);

/// Creates a llzk::felt::FeltType.
MLIR_CAPI_EXPORTED MlirType llzkFeltTypeGet(MlirContext context);

/// Returns true if the type is a FeltType.
LLZK_DECLARE_TYPE_ISA(FeltType);

#ifdef __cplusplus
}
#endif

#endif
