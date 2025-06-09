//===-- LLZK.h - C API for LLZK dialect ---------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// LLZK dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations, or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_LLZK_H
#define LLZK_C_DIALECT_LLZK_H

#include <llzk-c/Support.h>
#include <mlir-c/IR.h>

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(LLZK, llzk);

/// Creates a llzk::PublicAttr.
MLIR_CAPI_EXPORTED MlirAttribute llzkPublicAttrGet(MlirContext context);

/// Returns true if the attribute is a PublicAttr.
LLZK_DECLARE_ATTR_ISA(PublicAttr);

#ifdef __cplusplus
}
#endif

#endif
