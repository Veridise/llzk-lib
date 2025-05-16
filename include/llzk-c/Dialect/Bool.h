//===-- Bool.h - C API for Bool dialect -------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Bool dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_BOOL_H
#define LLZK_C_DIALECT_BOOL_H

#include "llzk-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

enum Cmp { Cmp_EQ, Cmp_NE, Cmp_LT, Cmp_LE, Cmp_GT, Cmp_GE };

#define DEFINE_C_API_STRUCT(name, storage)                                                         \
  struct name {                                                                                    \
    storage *ptr;                                                                                  \
  };                                                                                               \
  typedef struct name name

DEFINE_C_API_STRUCT(LlzkAndBoolOp, void);
DEFINE_C_API_STRUCT(LlzkOrBoolOp, void);
DEFINE_C_API_STRUCT(LlzkXorBoolOp, void);
DEFINE_C_API_STRUCT(LlzkNotBoolOp, void);
DEFINE_C_API_STRUCT(LlzkAssertOp, void);
DEFINE_C_API_STRUCT(LlzkCmpOp, void);

DEFINE_C_API_STRUCT(LlzkCmpPredicateAttr, const void);

#undef DEFINE_C_API_STRUCT

LLZK_DECLARE_CAPI_DIALECT_REGISTRATION(Bool, bool);

#ifdef __cplusplus
}
#endif

#endif
