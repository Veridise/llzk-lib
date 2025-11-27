//===-- Array.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Array.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/IR.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#include "../CAPITestBase.h"

// Include the auto-generated tests
#include "llzk/Dialect/Array/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Array/IR/Ops.capi.test.cpp.inc"
#include "llzk/Dialect/Array/IR/Types.capi.test.cpp.inc"

struct ArrayDialectTests : public CAPITest {
  MlirType test_array(MlirType elt, llvm::ArrayRef<int64_t> dims) const {
    return llzkArrayArrayTypeGetWithShape(elt, dims.size(), dims.data());
  }

  llvm::SmallVector<MlirOperation> create_n_ops(int64_t n_ops, MlirType elt_type) const {
    auto name = mlirStringRefCreateFromCString("arith.constant");
    auto attr_name = mlirIdentifierGet(context, mlirStringRefCreateFromCString("value"));
    auto location = mlirLocationUnknownGet(context);
    llvm::SmallVector<MlirOperation> result;
    for (int64_t n = 0; n < n_ops; n++) {
      auto attr = mlirNamedAttributeGet(attr_name, mlirIntegerAttrGet(elt_type, n));
      auto op_state = mlirOperationStateGet(name, location);
      mlirOperationStateAddResults(&op_state, 1, &elt_type);
      mlirOperationStateAddAttributes(&op_state, 1, &attr);

      auto created_op = mlirOperationCreate(&op_state);

      result.push_back(created_op);
    }
    return result;
  }
};

TEST_F(ArrayDialectTests, array_type_get) {
  auto size = createIndexAttribute(1);
  MlirAttribute dims[1] = {size};
  auto arr_type = llzkArrayArrayTypeGetWithDims(createIndexType(), 1, dims);
  EXPECT_NE(arr_type.ptr, (const void *)NULL);
}

TEST_F(ArrayDialectTests, type_is_a_array_type_pass) {
  auto size = createIndexAttribute(1);
  MlirAttribute dims[1] = {size};
  auto arr_type = llzkArrayArrayTypeGetWithDims(createIndexType(), 1, dims);
  EXPECT_NE(arr_type.ptr, (const void *)NULL);
  EXPECT_TRUE(llzkTypeIsAArrayArrayType(arr_type));
}

TEST_F(ArrayDialectTests, array_type_get_with_numeric_dims) {
  int64_t dims[2] = {1, 2};
  auto arr_type = llzkArrayArrayTypeGetWithShape(createIndexType(), 2, dims);
  EXPECT_NE(arr_type.ptr, (const void *)NULL);
}

TEST_F(ArrayDialectTests, array_type_get_element_type) {
  int64_t dims[2] = {1, 2};
  auto arr_type = llzkArrayArrayTypeGetWithShape(createIndexType(), 2, dims);
  EXPECT_NE(arr_type.ptr, (const void *)NULL);
  auto elt_type = llzkArrayArrayTypeGetElementType(arr_type);
  EXPECT_TRUE(mlirTypeEqual(createIndexType(), elt_type));
}

TEST_F(ArrayDialectTests, array_type_get_num_dims) {
  int64_t dims[2] = {1, 2};
  auto arr_type = llzkArrayArrayTypeGetWithShape(createIndexType(), 2, dims);
  EXPECT_NE(arr_type.ptr, (const void *)NULL);
  auto n_dims = llzkArrayArrayTypeGetDimensionSizesCount(arr_type);
  EXPECT_EQ(n_dims, 2);
}

TEST_F(ArrayDialectTests, array_type_get_dim) {
  int64_t dims[2] = {1, 2};
  auto arr_type = llzkArrayArrayTypeGetWithShape(createIndexType(), 2, dims);
  EXPECT_NE(arr_type.ptr, (const void *)NULL);
  auto out_dim = llzkArrayArrayTypeGetDimensionSizesAt(arr_type, 0);
  auto dim_as_attr = createIndexAttribute(dims[0]);
  EXPECT_TRUE(mlirAttributeEqual(out_dim, dim_as_attr));
}

struct CreateArrayOpBuildFuncHelper : public TestAnyBuildFuncHelper<ArrayDialectTests> {
  bool callIsA(MlirOperation op) override { return llzkOperationIsAArrayCreateArrayOp(op); }
};

TEST_F(ArrayDialectTests, create_array_op_build_with_values) {
  struct LocalHelper : CreateArrayOpBuildFuncHelper {
    llvm::SmallVector<MlirOperation> otherOps;

    MlirOperation callBuild(
        const ArrayDialectTests &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      int64_t dims[1] = {1};
      auto elt_type = testClass.createIndexType();
      auto test_type = testClass.test_array(elt_type, llvm::ArrayRef(dims, 1));
      this->otherOps = testClass.create_n_ops(1, elt_type);
      llvm::SmallVector<MlirValue> values;
      for (auto op : this->otherOps) {
        values.push_back(mlirOperationGetResult(op, 0));
      }
      return llzkArrayCreateArrayOpBuildWithValues(
          builder, location, test_type, values.size(), values.data()
      );
    }
    void doOtherChecks(MlirOperation) override {
      for (auto op : this->otherOps) {
        EXPECT_TRUE(mlirOperationVerify(op));
      }
    }
    ~LocalHelper() override {
      for (auto op : this->otherOps) {
        mlirOperationDestroy(op);
      }
    }
  } helper;
  helper.run(*this);
}

TEST_F(ArrayDialectTests, create_array_op_build_with_map_operands) {
  struct : CreateArrayOpBuildFuncHelper {
    MlirOperation callBuild(
        const ArrayDialectTests &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      int64_t dims[1] = {1};
      auto elt_type = testClass.createIndexType();
      auto test_type = testClass.test_array(elt_type, llvm::ArrayRef(dims, 1));
      auto dims_per_map = mlirDenseI32ArrayGet(testClass.context, 0, NULL);
      return llzkArrayCreateArrayOpBuildWithMapOperands(
          builder, location, test_type, 0, NULL, dims_per_map
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(ArrayDialectTests, create_array_op_build_with_map_operands_and_dims) {
  struct : CreateArrayOpBuildFuncHelper {
    MlirOperation callBuild(
        const ArrayDialectTests &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      int64_t dims[1] = {1};
      auto elt_type = testClass.createIndexType();
      auto test_type = testClass.test_array(elt_type, llvm::ArrayRef(dims, 1));
      return llzkArrayCreateArrayOpBuildWithMapOperandsAndDims(
          builder, location, test_type, 0, NULL, 0, NULL
      );
    }
  } helper;
  helper.run(*this);
}
