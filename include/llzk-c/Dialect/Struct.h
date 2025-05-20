//===-- Struct.h - C API for Struct dialect -------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Struct dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_STRUCT_H
#define LLZK_C_DIALECT_STRUCT_H

#include <stdint.h>

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Struct, llzk__component);

/// Creates a llzk::component::StructType.
MLIR_CAPI_EXPORTED MlirType llzkStructTypeGet(MlirContext, MlirAttribute);

/// Creates a llzk::component::StructType with an ArrayAttr as parameters.
MLIR_CAPI_EXPORTED
MlirType llzkStructTypeGetWithArrayAttr(MlirContext, MlirAttribute, MlirAttribute);

/// Creates a llzk::component::StructType with an array of parameters.
MLIR_CAPI_EXPORTED MlirType
llzkStructTypeGetWithAttrs(MlirContext, MlirAttribute, intptr_t, MlirAttribute const *);

/// Returns the fully qualified name of a llzk::component::StructType.
MLIR_CAPI_EXPORTED MlirAttribute llzkStructTypeGetName(MlirType);

/// Returns the parameter of a llzk::component::StructType as an ArrayAttr.
MLIR_CAPI_EXPORTED MlirAttribute llzkStructTypeGetParams(MlirType);

#ifdef __cplusplus
}
#endif

#endif
