//===-- Cast.h - C API for Cast dialect -------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Cast dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_CAST_H
#define LLZK_C_DIALECT_CAST_H

#include "llzk-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                                         \
  struct name {                                                                                    \
    storage *ptr;                                                                                  \
  };                                                                                               \
  typedef struct name name

DEFINE_C_API_STRUCT(LlzkIntToFeltOp, void);
DEFINE_C_API_STRUCT(LlzkFeltToIntOp, void);

#undef DEFINE_C_API_STRUCT

LLZK_DECLARE_CAPI_DIALECT_REGISTRATION(Cast, cast);

#ifdef __cplusplus
}
#endif

#endif
