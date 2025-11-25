//===-- Include.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Include.h"

#include "../CAPITestBase.h"

// Include the auto-generated tests
#include "llzk/Dialect/Include/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Include/IR/Ops.capi.test.cpp.inc"

TEST_F(CAPITest, llzk_include_op_create) {
  auto location = mlirLocationUnknownGet(context);
  auto op = llzkIncludeIncludeOpCreateInferredContext(
      location, mlirStringRefCreateFromCString("test"), mlirStringRefCreateFromCString("test.mlir")
  );

  EXPECT_NE(op.ptr, (void *)NULL);
  mlirOperationDestroy(op);
}
