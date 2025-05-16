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

#include "llzk-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                                         \
  struct name {                                                                                    \
    storage *ptr;                                                                                  \
  };                                                                                               \
  typedef struct name name

DEFINE_C_API_STRUCT(LlzkConstReadOp, void);
DEFINE_C_API_STRUCT(LlzkUnifiableCastOp, void);
DEFINE_C_API_STRUCT(LlzkApplyMapOp, void);

DEFINE_C_API_STRUCT(LlzkTypeVarType, const void);

#undef DEFINE_C_API_STRUCT

LLZK_DECLARE_CAPI_DIALECT_REGISTRATION(Polymorphic, polymorphic);

#ifdef __cplusplus
}
#endif

#endif
