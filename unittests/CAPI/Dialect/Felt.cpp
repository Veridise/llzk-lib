//===-- Felt.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Felt.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>

#include "../CAPITestBase.h"

TEST_F(CAPITest, mlir_get_dialect_handle_llzk_felt) { mlirGetDialectHandle__llzk__felt__(); }

TEST_F(CAPITest, llzk_felt_const_attr_get) {
  auto attr = llzkFeltConstAttrGet(context, 0);
  EXPECT_NE(attr.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_attribute_is_a_felt_const_attr_pass) {
  auto attr = llzkFeltConstAttrGet(context, 0);
  EXPECT_TRUE(llzkAttributeIsAFeltConstAttr(attr));
}

TEST_F(CAPITest, llzk_attribute_is_a_felt_const_attr_fail) {
  auto attr = mlirIntegerAttrGet(mlirIndexTypeGet(context), 0);
  EXPECT_TRUE(!llzkAttributeIsAFeltConstAttr(attr));
}

TEST_F(CAPITest, llzk_felt_type_get) {
  auto type = llzkFeltTypeGet(context);
  EXPECT_NE(type.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_type_is_a_felt_type_pass) {
  auto type = llzkFeltTypeGet(context);
  EXPECT_TRUE(llzkTypeIsAFeltType(type));
}

TEST_F(CAPITest, llzk_type_is_a_felt_type_fail) {
  auto type = mlirIndexTypeGet(context);
  EXPECT_TRUE(!llzkTypeIsAFeltType(type));
}
