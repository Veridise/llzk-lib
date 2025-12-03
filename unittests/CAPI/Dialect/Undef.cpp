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

TEST_F(CAPITest, mlir_get_dialect_handle_llzk_undef) {
  (void)mlirGetDialectHandle__llzk__undef__();
}

TEST_F(CAPITest, llzk_operation_is_a_undef_op_pass) {
  auto op_name = mlirStringRefCreateFromCString("undef.undef");
  auto state = mlirOperationStateGet(op_name, mlirLocationUnknownGet(context));
  auto t = llzkFeltTypeGet(context);
  mlirOperationStateAddResults(&state, 1, &t);

  auto op = mlirOperationCreate(&state);
  EXPECT_TRUE(llzkOperationIsAUndefOp(op));
}
