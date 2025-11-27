//===-- Poly.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Poly.h"

#include <mlir-c/BuiltinAttributes.h>

#include <llvm/ADT/SmallVector.h>

#include "../CAPITestBase.h"

// Include the auto-generated tests
#include "llzk/Dialect/Polymorphic/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Polymorphic/IR/Ops.capi.test.cpp.inc"
#include "llzk/Dialect/Polymorphic/IR/Types.capi.test.cpp.inc"

TEST_F(CAPITest, llzk_type_var_type_get) {
  auto t = llzkPolyTypeVarTypeGetFromStringRef(context, mlirStringRefCreateFromCString("T"));
  EXPECT_NE(t.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_type_is_a_type_var_type_pass) {
  auto t = llzkPolyTypeVarTypeGetFromStringRef(context, mlirStringRefCreateFromCString("T"));
  EXPECT_TRUE(llzkTypeIsAPolyTypeVarType(t));
}

TEST_F(CAPITest, llzk_type_var_type_get_from_attr) {
  auto s = mlirStringAttrGet(context, mlirStringRefCreateFromCString("T"));
  auto t = llzkPolyTypeVarTypeGetFromAttr(context, s);
  EXPECT_NE(t.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_type_var_type_get_name_ref) {
  auto s = mlirStringRefCreateFromCString("T");
  auto t = llzkPolyTypeVarTypeGetFromStringRef(context, s);
  EXPECT_NE(t.ptr, (void *)NULL);
  EXPECT_TRUE(mlirStringRefEqual(s, llzkPolyTypeVarTypeGetRefName(t)));
}

TEST_F(CAPITest, llzk_type_var_type_get_name) {
  auto s = mlirStringRefCreateFromCString("T");
  auto t = llzkPolyTypeVarTypeGetFromStringRef(context, s);
  auto sym = mlirFlatSymbolRefAttrGet(context, s);
  EXPECT_NE(t.ptr, (void *)NULL);
  EXPECT_TRUE(mlirAttributeEqual(sym, llzkPolyTypeVarTypeGetNameRef(t)));
}

struct ApplyMapOpBuildFuncHelper : public TestAnyBuildFuncHelper<CAPITest> {
  bool callIsA(MlirOperation op) override { return llzkOperationIsAPolyApplyMapOp(op); }
};

TEST_F(CAPITest, llzk_apply_map_op_build) {
  struct : ApplyMapOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      llvm::SmallVector<MlirAffineExpr> exprs({mlirAffineConstantExprGet(testClass.context, 1)});
      auto affine_map = mlirAffineMapGet(testClass.context, 0, 0, exprs.size(), exprs.data());
      auto affine_map_attr = mlirAffineMapAttrGet(affine_map);
      return llzkPolyApplyMapOpBuild(
          builder, location, affine_map_attr,
          MlirValueRange {
              .values = (const MlirValue *)NULL,
              .size = 0,
          }
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(CAPITest, llzk_apply_map_op_build_with_affine_map) {
  struct : ApplyMapOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      llvm::SmallVector<MlirAffineExpr> exprs({mlirAffineConstantExprGet(testClass.context, 1)});
      auto affine_map = mlirAffineMapGet(testClass.context, 0, 0, exprs.size(), exprs.data());
      return llzkPolyApplyMapOpBuildWithAffineMap(
          builder, location, affine_map,
          MlirValueRange {
              .values = (const MlirValue *)NULL,
              .size = 0,
          }
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(CAPITest, llzk_apply_map_op_build_with_affine_expr) {
  struct : ApplyMapOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      auto expr = mlirAffineConstantExprGet(testClass.context, 1);
      return llzkPolyApplyMapOpBuildWithAffineExpr(
          builder, location, expr,
          MlirValueRange {
              .values = (const MlirValue *)NULL,
              .size = 0,
          }
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(CAPITest, llzk_op_is_a_apply_map_op_pass) {
  struct : ApplyMapOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      auto expr = mlirAffineConstantExprGet(testClass.context, 1);
      return llzkPolyApplyMapOpBuildWithAffineExpr(
          builder, location, expr,
          MlirValueRange {
              .values = (const MlirValue *)NULL,
              .size = 0,
          }
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(CAPITest, llzk_apply_map_op_get_affine_map) {
  struct : ApplyMapOpBuildFuncHelper {
    MlirAffineMap affine_map;

    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      llvm::SmallVector<MlirAffineExpr> exprs({mlirAffineConstantExprGet(testClass.context, 1)});
      this->affine_map = mlirAffineMapGet(testClass.context, 0, 0, exprs.size(), exprs.data());
      return llzkPolyApplyMapOpBuildWithAffineMap(
          builder, location, this->affine_map,
          MlirValueRange {
              .values = (const MlirValue *)NULL,
              .size = 0,
          }
      );
    }
    void doOtherChecks(MlirOperation op) override {
      auto out_affine_map = llzkPolyApplyMapOpGetAffineMap(op);
      EXPECT_TRUE(mlirAffineMapEqual(this->affine_map, out_affine_map));
    }
  } helper;
  helper.run(*this);
}

TEST_F(CAPITest, llzk_apply_map_op_get_dim_operands) {
  struct : ApplyMapOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      llvm::SmallVector<MlirAffineExpr> exprs({mlirAffineConstantExprGet(testClass.context, 1)});
      auto affine_map = mlirAffineMapGet(testClass.context, 0, 0, exprs.size(), exprs.data());
      return llzkPolyApplyMapOpBuildWithAffineMap(
          builder, location, affine_map,
          MlirValueRange {
              .values = (const MlirValue *)NULL,
              .size = 0,
          }
      );
    }
    void doOtherChecks(MlirOperation op) override {
      auto n_dims = llzkPolyApplyMapOpGetNumDimOperands(op);
      llvm::SmallVector<MlirValue> dims(n_dims, MlirValue {.ptr = (void *)NULL});
      llzkPolyApplyMapOpGetDimOperands(op, dims.data());
      EXPECT_EQ(dims.size(), 0);
    }
  } helper;
  helper.run(*this);
}

TEST_F(CAPITest, llzk_apply_map_op_get_symbol_operands) {
  struct : ApplyMapOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      llvm::SmallVector<MlirAffineExpr> exprs = {mlirAffineConstantExprGet(testClass.context, 1)};
      auto affine_map = mlirAffineMapGet(testClass.context, 0, 0, exprs.size(), exprs.data());
      return llzkPolyApplyMapOpBuildWithAffineMap(
          builder, location, affine_map,
          MlirValueRange {
              .values = (const MlirValue *)NULL,
              .size = 0,
          }
      );
    }
    void doOtherChecks(MlirOperation op) override {
      auto n_syms = llzkPolyApplyMapOpGetNumSymbolOperands(op);
      llvm::SmallVector<MlirValue> syms(n_syms, {.ptr = (void *)NULL});
      llzkPolyApplyMapOpGetSymbolOperands(op, syms.data());
      EXPECT_EQ(syms.size(), 0);
    }
  } helper;
  helper.run(*this);
}
