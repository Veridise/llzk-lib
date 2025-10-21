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

class ArrayDialectTests : public CAPITest {
protected:
  MlirType indexType() { return mlirIndexTypeGet(context); }

  MlirType test_array(MlirType elt, llvm::ArrayRef<int64_t> dims) {
    return llzkArrayTypeGetWithNumericDims(elt, dims.size(), dims.data());
  }

  llvm::SmallVector<MlirOperation> create_n_ops(int64_t n_ops, MlirType elt_type) {
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

TEST_F(ArrayDialectTests, get_dialect_handle_llzk_array) {
  (void)mlirGetDialectHandle__llzk__array__();
}

TEST_F(ArrayDialectTests, array_type_get) {
  auto size = mlirIntegerAttrGet(indexType(), 1);
  MlirAttribute dims[1] = {size};
  auto arr_type = llzkArrayTypeGet(indexType(), 1, dims);
  EXPECT_NE(arr_type.ptr, (const void *)NULL);
}

TEST_F(ArrayDialectTests, type_isa_array_type_pass) {
  auto size = mlirIntegerAttrGet(indexType(), 1);
  MlirAttribute dims[1] = {size};
  auto arr_type = llzkArrayTypeGet(indexType(), 1, dims);
  EXPECT_NE(arr_type.ptr, (const void *)NULL);
  EXPECT_TRUE(llzkTypeIsAArrayType(arr_type));
}

TEST_F(ArrayDialectTests, type_isa_array_type_fail) {
  EXPECT_TRUE(!llzkTypeIsAArrayType(indexType()));
}

TEST_F(ArrayDialectTests, array_type_get_with_numeric_dims) {
  int64_t dims[2] = {1, 2};
  auto arr_type = llzkArrayTypeGetWithNumericDims(indexType(), 2, dims);
  EXPECT_NE(arr_type.ptr, (const void *)NULL);
}

TEST_F(ArrayDialectTests, array_type_get_element_type) {
  int64_t dims[2] = {1, 2};
  auto arr_type = llzkArrayTypeGetWithNumericDims(indexType(), 2, dims);
  EXPECT_NE(arr_type.ptr, (const void *)NULL);
  auto elt_type = llzkArrayTypeGetElementType(arr_type);
  EXPECT_TRUE(mlirTypeEqual(indexType(), elt_type));
}

TEST_F(ArrayDialectTests, array_type_get_num_dims) {
  int64_t dims[2] = {1, 2};
  auto arr_type = llzkArrayTypeGetWithNumericDims(indexType(), 2, dims);
  EXPECT_NE(arr_type.ptr, (const void *)NULL);
  auto n_dims = llzkArrayTypeGetNumDims(arr_type);
  EXPECT_EQ(n_dims, 2);
}

TEST_F(ArrayDialectTests, array_type_get_dim) {
  int64_t dims[2] = {1, 2};
  auto arr_type = llzkArrayTypeGetWithNumericDims(indexType(), 2, dims);
  EXPECT_NE(arr_type.ptr, (const void *)NULL);
  auto out_dim = llzkArrayTypeGetDim(arr_type, 0);
  auto dim_as_attr = mlirIntegerAttrGet(indexType(), dims[0]);
  EXPECT_TRUE(mlirAttributeEqual(out_dim, dim_as_attr));
}

TEST_F(ArrayDialectTests, create_array_op_build_with_values) {
  int64_t dims[1] = {1};
  auto elt_type = mlirIndexTypeGet(context);
  auto test_type = test_array(elt_type, llvm::ArrayRef(dims, 1));
  auto n_elements = 1;
  auto ops = create_n_ops(n_elements, elt_type);
  llvm::SmallVector<MlirValue> values;
  for (auto op : ops) {
    values.push_back(mlirOperationGetResult(op, 0));
  }
  auto builder = mlirOpBuilderCreate(context);
  auto location = mlirLocationUnknownGet(context);
  auto create_array_op =
      llzkCreateArrayOpBuildWithValues(builder, location, test_type, values.size(), values.data());
  for (auto op : ops) {
    EXPECT_TRUE(mlirOperationVerify(op));
  }

  EXPECT_TRUE(mlirOperationVerify(create_array_op));

  mlirOperationDestroy(create_array_op);
  for (auto op : ops) {
    mlirOperationDestroy(op);
  }
  mlirOpBuilderDestroy(builder);
}

TEST_F(ArrayDialectTests, create_array_op_build_with_map_operands) {
  int64_t dims[1] = {1};
  auto elt_type = mlirIndexTypeGet(context);
  auto test_type = test_array(elt_type, llvm::ArrayRef(dims, 1));

  auto builder = mlirOpBuilderCreate(context);
  auto location = mlirLocationUnknownGet(context);
  auto dims_per_map = mlirDenseI32ArrayGet(context, 0, NULL);

  auto op =
      llzkCreateArrayOpBuildWithMapOperands(builder, location, test_type, 0, NULL, dims_per_map);

  EXPECT_TRUE(mlirOperationVerify(op));
  mlirOperationDestroy(op);
  mlirOpBuilderDestroy(builder);
}

TEST_F(ArrayDialectTests, create_array_op_build_with_map_operands_and_dims) {
  int64_t dims[1] = {1};
  auto elt_type = mlirIndexTypeGet(context);
  auto test_type = test_array(elt_type, llvm::ArrayRef(dims, 1));

  auto builder = mlirOpBuilderCreate(context);
  auto location = mlirLocationUnknownGet(context);

  auto op =
      llzkCreateArrayOpBuildWithMapOperandsAndDims(builder, location, test_type, 0, NULL, 0, NULL);

  EXPECT_TRUE(mlirOperationVerify(op));
  mlirOperationDestroy(op);
  mlirOpBuilderDestroy(builder);
}
