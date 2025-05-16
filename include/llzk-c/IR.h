//===-- IR.h - General LLZK C API ---------------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares general utility functions and macros for the C API.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_IR_H
#define LLZK_C_IR_H

#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LLZK_DECLARE_CAPI_DIALECT_REGISTRATION(Name, Namespace)                                    \
  MLIR_CAPI_EXPORTED MlirDialectHandle llzkGetDialectHandle__##Namespace##__(void)

#ifdef __cplusplus
}
#endif

#endif
