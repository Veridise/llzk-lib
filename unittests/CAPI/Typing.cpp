//===-- Typing.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Typing.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/IR.h>

#include "CAPITestBase.h"

static bool test_callback1(MlirType, MlirType, void *) { return true; }

class TypingTest : public CAPITest {
protected:
  MlirType indexType() { return mlirIndexTypeGet(ctx); }
};

TEST_F(TypingTest, assert_valid_attr_for_param_of_type) {

  auto int_attr = mlirIntegerAttrGet(indexType(), 0);
  llzkAssertValidAttrForParamOfType(int_attr);
}

TEST_F(TypingTest, is_valid_type) { EXPECT_TRUE(llzkIsValidType(indexType())); }

TEST_F(TypingTest, is_valid_column_type) {

  auto null_op = MlirOperation {.ptr = NULL};
  EXPECT_TRUE(!llzkIsValidColumnType(indexType(), null_op));
}

TEST_F(TypingTest, is_valid_emit_eq_type) { EXPECT_TRUE(llzkIsValidEmitEqType(indexType())); }

TEST_F(TypingTest, is_valid_const_read_type) { EXPECT_TRUE(llzkIsValidConstReadType(indexType())); }

TEST_F(TypingTest, is_valid_array_elem_type) { EXPECT_TRUE(llzkIsValidArrayElemType(indexType())); }

TEST_F(TypingTest, is_valid_array_type) { EXPECT_TRUE(!llzkIsValidArrayType(indexType())); }

TEST_F(TypingTest, is_concrete_type) { EXPECT_TRUE(llzkIsConcreteType(indexType(), true)); }

TEST_F(TypingTest, is_signal_type) { EXPECT_TRUE(!llzkIsSignalType(indexType())); }

TEST_F(TypingTest, has_affine_map_attr) { EXPECT_TRUE(!llzkHasAffineMapAttr(indexType())); }

TEST_F(TypingTest, type_params_unify_empty) { EXPECT_TRUE(llzkTypeParamsUnify(0, NULL, 0, NULL)); }

TEST_F(TypingTest, type_params_unify_pass) {
  auto string_ref = mlirStringRefCreateFromCString("N");

  MlirAttribute lhs[1] = {mlirIntegerAttrGet(indexType(), 0)};
  MlirAttribute rhs[1] = {mlirFlatSymbolRefAttrGet(mlirAttributeGetContext(lhs[0]), string_ref)};
  EXPECT_TRUE(llzkTypeParamsUnify(1, lhs, 1, rhs));
}

TEST_F(TypingTest, type_params_unify_fail) {
  MlirAttribute lhs[1] = {mlirIntegerAttrGet(indexType(), 0)};
  MlirAttribute rhs[1] = {mlirIntegerAttrGet(indexType(), 1)};
  EXPECT_TRUE(!llzkTypeParamsUnify(1, lhs, 1, rhs));
}

TEST_F(TypingTest, array_attr_type_params_unify_empty) {
  MlirAttribute lhs = mlirArrayAttrGet(ctx, 0, NULL);
  MlirAttribute rhs = mlirArrayAttrGet(ctx, 0, NULL);
  EXPECT_TRUE(llzkArrayAttrTypeParamsUnify(lhs, rhs));
}

TEST_F(TypingTest, array_attr_type_params_unify_pass) {

  auto string_ref = mlirStringRefCreateFromCString("N");

  MlirAttribute lhs[1] = {mlirIntegerAttrGet(indexType(), 0)};
  auto lhsAttr = mlirArrayAttrGet(mlirAttributeGetContext(lhs[0]), 1, lhs);
  MlirAttribute rhs[1] = {mlirFlatSymbolRefAttrGet(mlirAttributeGetContext(*lhs), string_ref)};
  auto rhsAttr = mlirArrayAttrGet(mlirAttributeGetContext(*lhs), 1, rhs);
  EXPECT_TRUE(llzkArrayAttrTypeParamsUnify(lhsAttr, rhsAttr));
}

TEST_F(TypingTest, array_attr_type_params_unify_fail) {
  MlirAttribute lhs[1] = {mlirIntegerAttrGet(indexType(), 0)};

  auto lhsAttr = mlirArrayAttrGet(mlirAttributeGetContext(lhs[0]), 1, lhs);
  MlirAttribute rhs[1] = {mlirIntegerAttrGet(indexType(), 1)};

  auto rhsAttr = mlirArrayAttrGet(mlirAttributeGetContext(*lhs), 1, rhs);
  EXPECT_TRUE(!llzkArrayAttrTypeParamsUnify(lhsAttr, rhsAttr));
}

TEST_F(TypingTest, types_unify) { EXPECT_TRUE(llzkTypesUnify(indexType(), indexType(), 0, NULL)); }

TEST_F(TypingTest, is_more_concrete_unification) {
  EXPECT_TRUE(llzkIsMoreConcreteUnification(indexType(), indexType(), test_callback1, NULL));
}

TEST_F(TypingTest, force_int_attr_type) {
  auto in_attr = mlirIntegerAttrGet(mlirIntegerTypeGet(ctx, 64), 0);
  auto out_attr = llzkForceIntAttrType(in_attr);
  EXPECT_TRUE(!mlirAttributeEqual(in_attr, out_attr));
}

TEST_F(TypingTest, is_valid_global_type) { EXPECT_TRUE(llzkIsValidGlobalType(indexType())); }
