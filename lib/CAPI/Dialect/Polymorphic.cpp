//===-- Polymorphic.cpp - Polymorphic dialect C API impl --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llzk/CAPI/Builder.h>
#include <llzk/CAPI/Support.h>
#include <llzk/Dialect/Polymorphic/IR/Dialect.h>
#include <llzk/Dialect/Polymorphic/IR/Ops.h>
#include <llzk/Dialect/Polymorphic/IR/Types.h>
#include <llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h>

#include <mlir/CAPI/AffineExpr.h>
#include <mlir/CAPI/AffineMap.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>

#include <llzk-c/Dialect/Polymorphic.h>
#include <mlir-c/Pass.h>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"

static void registerLLZKPolymorphicTransformationPasses() {
  llzk::polymorphic::registerTransformationPasses();
}

using namespace llzk;
using namespace llzk::polymorphic;
using namespace mlir;

// Include impl for transformation passes
#include <llzk/Dialect/Polymorphic/Transforms/TransformationPasses.capi.cpp.inc>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(
    Polymorphic, llzk__polymorphic, llzk::polymorphic::PolymorphicDialect
)

//===----------------------------------------------------------------------===//
// TypeVarType
//===----------------------------------------------------------------------===//

MlirType llzkTypeVarTypeGet(MlirContext ctx, MlirStringRef name) {
  return wrap(TypeVarType::get(FlatSymbolRefAttr::get(StringAttr::get(unwrap(ctx), unwrap(name)))));
}

bool llzkTypeIsATypeVarType(MlirType type) { return mlir::isa<TypeVarType>(unwrap(type)); }

MlirType llzkTypeVarTypeGetFromAttr(MlirContext ctx, MlirAttribute attrWrapper) {
  auto attr = unwrap(attrWrapper);
  if (auto sym = mlir::dyn_cast<FlatSymbolRefAttr>(attr)) {

    return wrap(TypeVarType::get(sym));
  }
  return wrap(TypeVarType::get(FlatSymbolRefAttr::get(mlir::cast<StringAttr>(attr))));
}

MlirStringRef llzkTypeVarTypeGetNameRef(MlirType type) {
  return wrap(mlir::cast<TypeVarType>(unwrap(type)).getRefName());
}

MlirAttribute llzkTypeVarTypeGetName(MlirType type) {
  return wrap(mlir::cast<TypeVarType>(unwrap(type)).getNameRef());
}

//===----------------------------------------------------------------------===//
// ApplyMapOp
//===----------------------------------------------------------------------===//

LLZK_DEFINE_OP_BUILD_METHOD(ApplyMapOp, , MlirAttribute map, MlirValueRange mapOperands) {
  SmallVector<Value> mapOperandsSto;
  return wrap(
      create<ApplyMapOp>(
          builder, location, mlir::cast<AffineMapAttr>(unwrap(map)),
          ValueRange(unwrapList(mapOperands.size, mapOperands.values, mapOperandsSto))
      )
  );
}

LLZK_DEFINE_OP_BUILD_METHOD(
    ApplyMapOp, WithAffineMap, MlirAffineMap map, MlirValueRange mapOperands
) {
  SmallVector<Value> mapOperandsSto;
  return wrap(
      create<ApplyMapOp>(
          builder, location, unwrap(map),
          ValueRange(unwrapList(mapOperands.size, mapOperands.values, mapOperandsSto))
      )
  );
}

LLZK_DEFINE_OP_BUILD_METHOD(
    ApplyMapOp, WithAffineExpr, MlirAffineExpr expr, MlirValueRange mapOperands
) {
  SmallVector<Value> mapOperandsSto;
  return wrap(
      create<ApplyMapOp>(
          builder, location, unwrap(expr),
          ValueRange(unwrapList(mapOperands.size, mapOperands.values, mapOperandsSto))
      )
  );
}

bool llzkOperationIsAApplyMapOp(MlirOperation op) { return mlir::isa<ApplyMapOp>(unwrap(op)); }

/// Returns the affine map associated with the op.
MlirAffineMap llzkApplyMapOpGetAffineMap(MlirOperation op) {
  return wrap(mlir::cast<ApplyMapOp>(unwrap(op)).getAffineMap());
}

/// Returns the operands that correspond to dimensions in the affine map.
MlirValueRange llzkApplyMapOpGetDimOperands(MlirOperation op) {
  return wrap(mlir::cast<ApplyMapOp>(unwrap(op)).getDimOperands());
}

/// Returns the operands that correspond to symbols in the affine map.
MlirValueRange llzkApplyMapOpGetSymbolOperands(MlirOperation op) {
  return wrap(mlir::cast<ApplyMapOp>(unwrap(op)).getSymbolOperands());
}
