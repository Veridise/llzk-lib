//===-- Global.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Global.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

#include <llvm/ADT/SmallVector.h>

#include "../CAPITestBase.h"

// Include the auto-generated tests
#include "llzk/Dialect/Global/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Global/IR/Ops.capi.test.cpp.inc"

static MlirNamedAttribute named_attr(const char *s, MlirAttribute attr) {
  return mlirNamedAttributeGet(
      mlirIdentifierGet(mlirAttributeGetContext(attr), mlirStringRefCreateFromCString(s)), attr
  );
}

static MlirOperation create_global_def_op(
    MlirContext ctx, const char *name, bool constant, MlirType type,
    std::optional<MlirAttribute> initial_value
) {
  auto sym_name = mlirFlatSymbolRefAttrGet(ctx, mlirStringRefCreateFromCString(name));
  auto attrs = llvm::SmallVector<MlirNamedAttribute>({
      named_attr("sym_name", sym_name),
      named_attr("type", mlirTypeAttrGet(type)),
  });
  if (constant) {
    attrs.push_back(named_attr("constant", mlirUnitAttrGet(ctx)));
  }
  if (initial_value.has_value()) {
    attrs.push_back(named_attr("initial_value", *initial_value));
  }
  auto op_name = mlirStringRefCreateFromCString("global.def");
  auto state = mlirOperationStateGet(op_name, mlirLocationUnknownGet(ctx));
  mlirOperationStateAddAttributes(&state, attrs.size(), attrs.data());

  return mlirOperationCreate(&state);
}

TEST_F(CAPITest, llzk_operation_is_a_global_def_op_pass) {
  auto op = create_global_def_op(context, "G", false, createIndexType(), std::nullopt);
  EXPECT_NE(op.ptr, (void *)NULL);
  EXPECT_TRUE(llzkOperationIsAGlobalGlobalDefOp(op));
  mlirOperationDestroy(op);
}

TEST_F(CAPITest, llzk_global_def_op_is_constant_1) {
  auto op = create_global_def_op(context, "G", false, createIndexType(), std::nullopt);
  EXPECT_NE(op.ptr, (void *)NULL);
  EXPECT_TRUE(!llzkGlobalGlobalDefOpIsConstant(op));
  mlirOperationDestroy(op);
}

TEST_F(CAPITest, llzk_global_def_op_is_constant_2) {
  auto op = create_global_def_op(context, "G", true, createIndexType(), createIndexAttribute(1));
  EXPECT_NE(op.ptr, (void *)NULL);
  EXPECT_TRUE(llzkGlobalGlobalDefOpIsConstant(op));
  mlirOperationDestroy(op);
}
