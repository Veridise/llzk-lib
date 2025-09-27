//===-- TypeHelperTest.cpp - Unit tests for symbol utilities ----*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/BuiltinTypeInterfaces.h>

#include <gtest/gtest.h>

#include "../LLZKTestBase.h"

using namespace mlir;
using namespace llzk;
using namespace llzk::array;
using namespace llzk::component;

class TypeHelperTests : public LLZKTest {
protected:
  TypeHelperTests() : LLZKTest() {}
};

TEST_F(TypeHelperTests, test_arrayTypesUnify_withDynamic_1) {
  IndexType tyIndex = IndexType::get(&ctx);
  ArrayType a = ArrayType::get(tyIndex, {2, ShapedType::kDynamic});
  ArrayType b = ArrayType::get(tyIndex, {2, 5});
  ASSERT_TRUE(arrayTypesUnify(a, b));
}

TEST_F(TypeHelperTests, test_arrayTypesUnify_withDynamic_2) {
  IndexType tyIndex = IndexType::get(&ctx);
  ArrayType a = ArrayType::get(tyIndex, {2, ShapedType::kDynamic});
  ArrayType b = ArrayType::get(tyIndex, {ShapedType::kDynamic, 5});
  ASSERT_TRUE(arrayTypesUnify(a, b));
}

TEST_F(TypeHelperTests, test_structTypesUnify) {
  IndexType tyIndex = IndexType::get(&ctx);
  Attribute i1 = IntegerAttr::get(tyIndex, 128);
  Attribute i2 = IntegerAttr::get(tyIndex, ShapedType::kDynamic);
  StructType a = StructType::get(FlatSymbolRefAttr::get(&ctx, "TheName"), ArrayRef {i1});
  StructType b = StructType::get(FlatSymbolRefAttr::get(&ctx, "TheName"), ArrayRef {i2});
  // `false` because StructType does not allow `kDynamic`
  ASSERT_FALSE(structTypesUnify(a, b));
}

TEST_F(TypeHelperTests, test_forceIntType) {
  // create an 8-bit IntegerAttr
  IntegerAttr a = IntegerAttr::get(IntegerType::get(&ctx, 8), 42);
  // Force IndexType on it without changing the value
  IntegerAttr b = forceIntType(a);
  ASSERT_TRUE(llvm::isa<IndexType>(b.getType()));
  ASSERT_EQ(b.getValue().getZExtValue(), 42);
}
