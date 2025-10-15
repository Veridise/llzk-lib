//===-- Undef.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Undef.h"

#include "llzk-c/Dialect/Felt.h"

#include <mlir-c/IR.h>

#include "../CAPITestBase.h"

// Include the auto-generated tests
#include "llzk/Dialect/Undef/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Undef/IR/Ops.capi.test.cpp.inc"

TEST_F(CAPITest, llzkOperationIsAUndefOp) {
  auto op_name = mlirStringRefCreateFromCString("undef.undef");
  auto state = mlirOperationStateGet(op_name, mlirLocationUnknownGet(context));
  auto t = llzkFeltFeltTypeGet(context);
  mlirOperationStateAddResults(&state, 1, &t);

  auto op = mlirOperationCreate(&state);
  EXPECT_TRUE(llzkOperationIsAUndefOp(op));
}
