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

#include <llzk/Dialect/Polymorphic/Transforms/TransformationPasses.capi.h.inc>

#include <llzk-c/Support.h>
#include <mlir-c/AffineExpr.h>
#include <mlir-c/AffineMap.h>
#include <mlir-c/IR.h>

#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Polymorphic, llzk__polymorphic);

//===----------------------------------------------------------------------===//
// TypeVarType
//===----------------------------------------------------------------------===//

/// Creates a llzk::polymorphic::TypeVarType.
MLIR_CAPI_EXPORTED MlirType llzkTypeVarTypeGet(MlirContext, MlirStringRef);

/// Returns true if the type is a TypeVarType.
LLZK_DECLARE_TYPE_ISA(TypeVarType);

/// Creates a llzk::polymorphic::TypeVarType from either a StringAttr or a FlatSymbolRefAttr.
MLIR_CAPI_EXPORTED MlirType llzkTypeVarTypeGetFromAttr(MlirContext, MlirAttribute);

/// Returns the var name of the TypeVarType as a StringRef.
MLIR_CAPI_EXPORTED MlirStringRef llzkTypeVarTypeGetNameRef(MlirType);

/// Returns the var name of the TypeVarType as a FlatSymbolRefAttr.
MLIR_CAPI_EXPORTED MlirAttribute llzkTypeVarTypeGetName(MlirType);

//===----------------------------------------------------------------------===//
// ApplyMapOp
//===----------------------------------------------------------------------===//

/// Creates an ApplyMapOp with the given attribute that has to be of type AffineMapAttr.
LLZK_DECLARE_OP_BUILD_METHOD(ApplyMapOp, , MlirAttribute, MlirValueRange);

/// Creates an ApplyMapOp with the given affine map.
LLZK_DECLARE_OP_BUILD_METHOD(ApplyMapOp, WithAffineMap, MlirAffineMap, MlirValueRange);

/// Creates an ApplyMapOp with the given affine expression.
LLZK_DECLARE_OP_BUILD_METHOD(ApplyMapOp, WithAffineExpr, MlirAffineExpr, MlirValueRange);

/// Returns true if the op is an ApplyMapOp.
LLZK_DECLARE_OP_ISA(ApplyMapOp);

/// Returns the affine map associated with the op.
MLIR_CAPI_EXPORTED MlirAffineMap llzkApplyMapOpGetAffineMap(MlirOperation);

/// Returns the operands that correspond to dimensions in the affine map.
MLIR_CAPI_EXPORTED MlirValueRange llzkApplyMapOpGetDimOperands(MlirOperation);

/// Returns the operands that correspond to symbols in the affine map.
MLIR_CAPI_EXPORTED MlirValueRange llzkApplyMapOpGetSymbolOperands(MlirOperation);

#ifdef __cplusplus
}
#endif

#endif
