//===-- Support.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Support.h"

#include "CAPITestBase.h"

TEST_F(CAPITest, MlirOperationReplaceUsesOfWith) {
  // Create two constant operations that produce values
  MlirOperation const1 = createIndexOperation();
  MlirOperation const2 = createIndexOperation();
  MlirValue value1 = mlirOperationGetResult(const1, 0);
  MlirValue value2 = mlirOperationGetResult(const2, 0);

  // Create an operation that uses value1: `arith.addi value1, value1 : index`
  MlirType indexType = createIndexType();
  MlirOperationState addState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("arith.addi"), mlirLocationUnknownGet(context)
  );
  mlirOperationStateAddResults(&addState, 1, &indexType);
  MlirValue operands[2] = {value1, value1};
  mlirOperationStateAddOperands(&addState, 2, operands);
  MlirOperation addOp = mlirOperationCreate(&addState);

  // Verify that the add operation uses value1 twice
  EXPECT_EQ(mlirOperationGetNumOperands(addOp), 2);
  EXPECT_TRUE(mlirValueEqual(mlirOperationGetOperand(addOp, 0), value1));
  EXPECT_TRUE(mlirValueEqual(mlirOperationGetOperand(addOp, 1), value1));

  // Replace uses of value1 with value2 inside the add operation
  mlirOperationReplaceUsesOfWith(addOp, value1, value2);

  // Verify that both operands now use value2
  EXPECT_TRUE(mlirValueEqual(mlirOperationGetOperand(addOp, 0), value2));
  EXPECT_TRUE(mlirValueEqual(mlirOperationGetOperand(addOp, 1), value2));

  // Clean up
  mlirOperationDestroy(addOp);
  mlirOperationDestroy(const2);
  mlirOperationDestroy(const1);
}
