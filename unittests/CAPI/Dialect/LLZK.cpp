//===-- LLZK.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/LLZK.h"

#include "../CAPITestBase.h"

TEST_F(CAPITest, mlir_get_dialect_handle_llzk) { (void)mlirGetDialectHandle__llzk__(); }

TEST_F(CAPITest, llzk_public_attr_get) {
  auto attr = llzkPublicAttrGet(context);
  EXPECT_NE(attr.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_attribute_is_a_public_attr_pass) {
  auto attr = llzkPublicAttrGet(context);
  EXPECT_TRUE(llzkAttributeIsAPublicAttr(attr));
}
