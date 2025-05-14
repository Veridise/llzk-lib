//===-- Felt.h - C API for Felt dialect -------------------------*- C -*-===//
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
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_FELT_H
#define LLZK_C_DIALECT_FELT_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                                         \
  struct name {                                                                                    \
    storage *ptr;                                                                                  \
  };                                                                                               \
  typedef struct name name

DEFINE_C_API_STRUCT(LLzkFeltConstantOp, void);
DEFINE_C_API_STRUCT(LLzkFeltNonDetOp, void);
DEFINE_C_API_STRUCT(LLzkAddFeltOp, void);
DEFINE_C_API_STRUCT(LLzkSubFeltOp, void);
DEFINE_C_API_STRUCT(LLzkMulFeltOp, void);
DEFINE_C_API_STRUCT(LLzkDivFeltOp, void);
DEFINE_C_API_STRUCT(LLzkModFeltOp, void);
DEFINE_C_API_STRUCT(LLzkNegFeltOp, void);
DEFINE_C_API_STRUCT(LLzkInvFeltOp, void);
DEFINE_C_API_STRUCT(LLzkAndFeltOp, void);
DEFINE_C_API_STRUCT(LLzkOrFeltOp, void);
DEFINE_C_API_STRUCT(LLzkXorFeltOp, void);
DEFINE_C_API_STRUCT(LLzkNotFeltOp, void);
DEFINE_C_API_STRUCT(LLzkShlFeltOp, void);
DEFINE_C_API_STRUCT(LLzkShrFeltOp, void);

DEFINE_C_API_STRUCT(LLzkFeltConstAttr, const void);
DEFINE_C_API_STRUCT(LLzkFeltType, const void);

#undef DEFINE_C_API_STRUCT

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Felt, felt);

#ifdef __cplusplus
}
#endif

#endif
