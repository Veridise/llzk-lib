//===-- Boolean.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Bool.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>

#include "../CAPITestBase.h"

// Include the auto-generated tests
#include "llzk/Dialect/Bool/IR/Attrs.capi.test.cpp.inc"
#include "llzk/Dialect/Bool/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Bool/IR/Enums.capi.test.cpp.inc"
#include "llzk/Dialect/Bool/IR/Ops.capi.test.cpp.inc"

class CmpAttrTest : public CAPITest,
                    public testing::WithParamInterface<LlzkBoolFeltCmpPredicate> {};

TEST_P(CmpAttrTest, llzk_felt_cmp_predicate_attr_get) {
  auto attr = llzkBoolFeltCmpPredicateAttrGet(context, GetParam());
  EXPECT_NE(attr.ptr, (void *)NULL);
}

INSTANTIATE_TEST_SUITE_P(
    AllLlzkBoolFeltCmpPredicateValues, CmpAttrTest,
    testing::Values(
        LlzkBoolFeltCmpPredicate_EQ, LlzkBoolFeltCmpPredicate_LE, LlzkBoolFeltCmpPredicate_LT,
        LlzkBoolFeltCmpPredicate_GE, LlzkBoolFeltCmpPredicate_GT, LlzkBoolFeltCmpPredicate_NE
    )
);

TEST_F(CAPITest, llzk_attribute_is_a_felt_cmp_predicate_attr_pass) {
  auto attr = llzkBoolFeltCmpPredicateAttrGet(context, LlzkBoolFeltCmpPredicate_EQ);
  EXPECT_TRUE(llzkAttributeIsAFeltCmpPredicateAttr(attr));
}
