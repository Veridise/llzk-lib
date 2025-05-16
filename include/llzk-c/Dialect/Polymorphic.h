//===-- Polymorphic.h - C API for Polymorphic dialect -------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Polymorphic dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_POLYMORPHIC_H
#define LLZK_C_DIALECT_POLYMORPHIC_H

#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.capi.h.inc"

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Polymorphic, llzk__polymorphic);

/// Creates a llzk::polymorphic::TypeVarType.
MLIR_CAPI_EXPORTED MlirType llzkTypeVarTypeGet(MlirContext, MlirStringRef);

#ifdef __cplusplus
}
#endif

#endif
