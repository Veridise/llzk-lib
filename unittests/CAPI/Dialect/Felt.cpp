//===-- Felt.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Felt.h"
#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Support.h>

#include <llvm/ADT/APInt.h>

#include "../CAPITestBase.h"

// Include the auto-generated tests
#include "llzk/Dialect/Felt/IR/Attrs.capi.test.cpp.inc"
#include "llzk/Dialect/Felt/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Felt/IR/Ops.capi.test.cpp.inc"
#include "llzk/Dialect/Felt/IR/Types.capi.test.cpp.inc"

TEST_F(CAPITest, llzk_felt_const_attr_get) {
  auto attr = llzkFeltFeltConstAttrGet(context, 0);
  EXPECT_NE(attr.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzkFeltFeltConstAttrGetWithBits) {
  constexpr auto BITS = 128;
  auto attr = llzkFeltFeltConstAttrGetWithBits(context, BITS, 0);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto cxx_attr = llvm::dyn_cast<llzk::felt::FeltConstAttr>(unwrap(attr));
  EXPECT_TRUE(cxx_attr);
  auto value = cxx_attr.getValue();
  EXPECT_EQ(value.getBitWidth(), BITS);
  EXPECT_EQ(value.getZExtValue(), 0);
}

TEST_F(CAPITest, llzkFeltFeltConstAttrGetFromString) {
  constexpr auto BITS = 64;
  auto str = MlirStringRef {.data = "123", .length = 3};
  auto attr = llzkFeltFeltConstAttrGetFromString(context, BITS, str);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto expected = llzk::felt::FeltConstAttr::get(
      unwrap(context), llvm::APInt(BITS, llvm::StringRef("123", 3), 10)
  );
  EXPECT_EQ(unwrap(attr), expected);
}

TEST_F(CAPITest, llzkFeltFeltConstAttrGetFromParts) {
  constexpr auto BITS = 254;
  const uint64_t parts[] = {10, 20, 30, 40};
  auto attr = llzkFeltFeltConstAttrGetFromParts(context, BITS, parts, 4);
  EXPECT_NE(attr.ptr, (void *)NULL);
  auto expected =
      llzk::felt::FeltConstAttr::get(unwrap(context), llvm::APInt(BITS, llvm::ArrayRef(parts, 4)));
  EXPECT_EQ(unwrap(attr), expected);
}

TEST_F(CAPITest, llzk_attribute_is_a_felt_const_attr_pass) {
  auto attr = llzkFeltFeltConstAttrGet(context, 0);
  EXPECT_TRUE(llzkAttributeIsAFeltConstAttr(attr));
}

TEST_F(CAPITest, llzk_felt_type_get) {
  auto type = llzkFeltFeltTypeGet(context);
  EXPECT_NE(type.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_type_is_a_felt_type_pass) {
  auto type = llzkFeltFeltTypeGet(context);
  EXPECT_TRUE(llzkTypeIsAFeltType(type));
}
