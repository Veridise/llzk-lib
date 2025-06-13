//===-- Struct.cpp - Struct dialect C API implementation --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llzk/CAPI/Builder.h>
#include <llzk/CAPI/Support.h>
#include <llzk/Dialect/Function/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Dialect.h>
#include <llzk/Dialect/Struct/IR/Ops.h>
#include <llzk/Dialect/Struct/IR/Types.h>
#include <llzk/Util/TypeHelper.h>

#include <mlir/CAPI/AffineMap.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Support.h>
#include <mlir/CAPI/Wrap.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <llvm/ADT/STLExtras.h>

#include <llzk-c/Dialect/Struct.h>
#include <mlir-c/Support.h>

using namespace llzk;
using namespace mlir;
using namespace llzk::component;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Struct, llzk__component, StructDialect)

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

MlirType llzkStructTypeGet(MlirAttribute name) {
  return wrap(StructType::get(mlir::cast<SymbolRefAttr>(unwrap(name))));
}

MlirType llzkStructTypeGetWithArrayAttr(MlirAttribute name, MlirAttribute params) {
  return wrap(StructType::get(
      mlir::cast<SymbolRefAttr>(unwrap(name)), mlir::cast<ArrayAttr>(unwrap(params))
  ));
}

MlirType
llzkStructTypeGetWithAttrs(MlirAttribute name, intptr_t numParams, MlirAttribute const *params) {
  SmallVector<Attribute> paramsSto;
  return wrap(StructType::get(
      mlir::cast<SymbolRefAttr>(unwrap(name)), unwrapList(numParams, params, paramsSto)
  ));
}

bool llzkTypeIsAStructType(MlirType type) { return mlir::isa<StructType>(unwrap(type)); }

MlirAttribute llzkStructTypeGetName(MlirType type) {
  return wrap(mlir::cast<StructType>(unwrap(type)).getNameRef());
}

MlirAttribute llzkStructTypeGetParams(MlirType type) {
  return wrap(mlir::cast<StructType>(unwrap(type)).getParams());
}

//===----------------------------------------------------------------------===//
// StructDefOp
//===----------------------------------------------------------------------===//

bool llzkOperationIsAStructDefOp(MlirOperation op) { return mlir::isa<StructDefOp>(unwrap(op)); }

MlirType llzkStructDefOpGetType(MlirOperation op) {
  return wrap(mlir::cast<StructDefOp>(unwrap(op)).getType());
}

MlirType llzkStructDefOpGetTypeWithParams(MlirOperation op, MlirAttribute attr) {

  return wrap(mlir::cast<StructDefOp>(unwrap(op)).getType(mlir::cast<ArrayAttr>(unwrap(attr))));
}

MlirOperation llzkStructDefOpGetFieldDef(MlirOperation op, MlirStringRef name) {
  Builder builder(unwrap(op)->getContext());
  return wrap(mlir::cast<StructDefOp>(unwrap(op)).getFieldDef(builder.getStringAttr(unwrap(name))));
}

void llzkStructDefOpGetFieldDefs(MlirOperation op, MlirOperation *dst) {
  for (auto [offset, field] : llvm::enumerate(mlir::cast<StructDefOp>(unwrap(op)).getFieldDefs())) {
    dst[offset] = wrap(field);
  }
}

intptr_t llzkStructDefOpGetNumFieldDefs(MlirOperation op) {
  return static_cast<intptr_t>(mlir::cast<StructDefOp>(unwrap(op)).getFieldDefs().size());
}

MlirLogicalResult llzkStructDefOpGetHasColumns(MlirOperation op) {
  return wrap(mlir::cast<StructDefOp>(unwrap(op)).hasColumns());
}

MlirOperation llzkStructDefOpGetComputeFuncOp(MlirOperation op) {
  return wrap(mlir::cast<StructDefOp>(unwrap(op)).getComputeFuncOp());
}

MlirOperation llzkStructDefOpGetConstrainFuncOp(MlirOperation op) {
  return wrap(mlir::cast<StructDefOp>(unwrap(op)).getConstrainFuncOp());
}

const char *
llzkStructDefOpGetHeaderString(MlirOperation op, intptr_t *strSize, char *(*alloc_string)(size_t)) {
  auto header = mlir::cast<StructDefOp>(unwrap(op)).getHeaderString();
  *strSize = static_cast<intptr_t>(header.size()) + 1; // Plus one because it's a C string.
  char *dst = alloc_string(*strSize);
  dst[header.size()] = 0;
  memcpy(dst, header.data(), header.size());
  return dst;
}

bool llzkStructDefOpGetHasParamName(MlirOperation op, MlirStringRef name) {
  Builder builder(unwrap(op)->getContext());
  return mlir::cast<StructDefOp>(unwrap(op)).hasParamNamed(builder.getStringAttr(unwrap(name)));
}

MlirAttribute llzkStructDefOpGetFullyQualifiedName(MlirOperation op) {
  return wrap(mlir::cast<StructDefOp>(unwrap(op)).getFullyQualifiedName());
}

bool llzkStructDefOpGetIsMainComponent(MlirOperation op) {
  return mlir::cast<StructDefOp>(unwrap(op)).isMainComponent();
}

//===----------------------------------------------------------------------===//
// FieldDefOp
//===----------------------------------------------------------------------===//

bool llzkOperationIsAFieldDefOp(MlirOperation op) { return mlir::isa<FieldDefOp>(unwrap(op)); }

bool llzkFieldDefOpGetHasPublicAttr(MlirOperation op) {
  return mlir::cast<FieldDefOp>(unwrap(op)).hasPublicAttr();
}

void llzkFieldDefOpSetPublicAttr(MlirOperation op, bool value) {
  mlir::cast<FieldDefOp>(unwrap(op)).setPublicAttr(value);
}

//===----------------------------------------------------------------------===//
// FieldReadOp
//===----------------------------------------------------------------------===//

LLZK_DEFINE_OP_BUILD_METHOD(
    FieldReadOp, MlirType fieldType, MlirValue component, MlirStringRef name
) {
  return wrap(create<FieldReadOp>(
      builder, location, unwrap(fieldType), unwrap(component),
      unwrap(builder)->getStringAttr(unwrap(name))
  ));
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    FieldReadOp, WithAffineMapDistance, MlirType fieldType, MlirValue component, MlirStringRef name,
    MlirAffineMap map, MlirValueRange mapOperands, int32_t numDimsPerMap
) {
  SmallVector<Value> mapOperandsSto;
  auto nameAttr = unwrap(builder)->getStringAttr(unwrap(name));
  auto mapAttr = AffineMapAttr::get(unwrap(map));
  return wrap(create<FieldReadOp>(
      builder, location, unwrap(fieldType), unwrap(component), nameAttr, mapAttr,
      unwrapList(mapOperands.size, mapOperands.values, mapOperandsSto), numDimsPerMap
  ));
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    FieldReadOp, WithConstParamDistance, MlirType fieldType, MlirValue component,
    MlirStringRef name, MlirStringRef symbol
) {
  auto nameAttr = unwrap(builder)->getStringAttr(unwrap(name));
  return wrap(create<FieldReadOp>(
      builder, location, unwrap(fieldType), unwrap(component), nameAttr,
      FlatSymbolRefAttr::get(unwrap(builder)->getStringAttr(unwrap(symbol)))
  ));
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    FieldReadOp, WithLiteralDistance, MlirType fieldType, MlirValue component, MlirStringRef name,
    int64_t distance
) {
  auto nameAttr = unwrap(builder)->getStringAttr(unwrap(name));
  return wrap(create<FieldReadOp>(
      builder, location, unwrap(fieldType), unwrap(component), nameAttr,
      unwrap(builder)->getI64IntegerAttr(distance)
  ));
}
