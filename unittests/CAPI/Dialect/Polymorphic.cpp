//===-- Polymorphic.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Polymorphic.h"

#include <mlir-c/BuiltinAttributes.h>

#include <llvm/ADT/SmallVector.h>

#include "../CAPITestBase.h"
TEST_F(CAPITest, mlir_get_dialect_handle_llzk_polymorphic) {
  { mlirGetDialectHandle__llzk__polymorphic__(); }
}

TEST_F(CAPITest, llzk_type_var_type_get) {
  {
    auto t = llzkTypeVarTypeGet(ctx, mlirStringRefCreateFromCString("T"));
    EXPECT_NE(t.ptr, (void *)NULL);
  }
}

TEST_F(CAPITest, llzk_type_is_a_type_var_type) {
  {
    auto t = llzkTypeVarTypeGet(ctx, mlirStringRefCreateFromCString("T"));
    EXPECT_TRUE(llzkTypeIsATypeVarType(t));
  }
}

TEST_F(CAPITest, llzk_type_var_type_get_from_attr) {
  {
    auto s = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("T"));
    auto t = llzkTypeVarTypeGetFromAttr(ctx, s);
    EXPECT_NE(t.ptr, (void *)NULL);
  }
}

TEST_F(CAPITest, llzk_type_var_type_get_name_ref) {
  {
    auto s = mlirStringRefCreateFromCString("T");
    auto t = llzkTypeVarTypeGet(ctx, s);
    EXPECT_NE(t.ptr, (void *)NULL);
    EXPECT_TRUE(mlirStringRefEqual(s, llzkTypeVarTypeGetNameRef(t)));
  }
}

TEST_F(CAPITest, llzk_type_var_type_get_name) {
  {
    auto s = mlirStringRefCreateFromCString("T");
    auto t = llzkTypeVarTypeGet(ctx, s);
    auto sym = mlirFlatSymbolRefAttrGet(ctx, s);
    EXPECT_NE(t.ptr, (void *)NULL);
    EXPECT_TRUE(mlirAttributeEqual(sym, llzkTypeVarTypeGetName(t)));
  }
}

TEST_F(CAPITest, llzk_apply_map_op_build) {
  {
    auto builder = mlirOpBuilderCreate(ctx);
    auto location = mlirLocationUnknownGet(ctx);
    llvm::SmallVector<MlirAffineExpr> exprs({mlirAffineConstantExprGet(ctx, 1)});
    auto affine_map = mlirAffineMapGet(ctx, 0, 0, exprs.size(), exprs.data());
    auto affine_map_attr = mlirAffineMapAttrGet(affine_map);
    auto op = llzkApplyMapOpBuild(
        builder, location, affine_map_attr,
        MlirValueRange {
            .values = (const MlirValue *)NULL,
            .size = 0,
        }
    );
    EXPECT_NE(op.ptr, (void *)NULL);
    EXPECT_TRUE(mlirOperationVerify(op));
    mlirOperationDestroy(op);
    mlirOpBuilderDestroy(builder);
  }
}

TEST_F(CAPITest, llzk_apply_map_op_build_with_affine_map) {
  {
    auto builder = mlirOpBuilderCreate(ctx);
    auto location = mlirLocationUnknownGet(ctx);
    llvm::SmallVector<MlirAffineExpr> exprs({mlirAffineConstantExprGet(ctx, 1)});
    auto affine_map = mlirAffineMapGet(ctx, 0, 0, exprs.size(), exprs.data());
    auto op = llzkApplyMapOpBuildWithAffineMap(
        builder, location, affine_map,
        MlirValueRange {
            .values = (const MlirValue *)NULL,
            .size = 0,
        }
    );
    EXPECT_NE(op.ptr, (void *)NULL);
    EXPECT_TRUE(mlirOperationVerify(op));
    mlirOperationDestroy(op);
    mlirOpBuilderDestroy(builder);
  }
}

TEST_F(CAPITest, llzk_apply_map_op_build_with_affine_expr) {
  {
    auto builder = mlirOpBuilderCreate(ctx);
    auto location = mlirLocationUnknownGet(ctx);
    auto expr = mlirAffineConstantExprGet(ctx, 1);
    auto op = llzkApplyMapOpBuildWithAffineExpr(
        builder, location, expr,
        MlirValueRange {
            .values = (const MlirValue *)NULL,
            .size = 0,
        }
    );
    EXPECT_NE(op.ptr, (void *)NULL);
    EXPECT_TRUE(mlirOperationVerify(op));
    mlirOperationDestroy(op);
    mlirOpBuilderDestroy(builder);
  }
}

TEST_F(CAPITest, llzk_op_is_a_apply_map_op) {
  {
    auto builder = mlirOpBuilderCreate(ctx);
    auto location = mlirLocationUnknownGet(ctx);
    auto expr = mlirAffineConstantExprGet(ctx, 1);
    auto op = llzkApplyMapOpBuildWithAffineExpr(
        builder, location, expr,
        MlirValueRange {
            .values = (const MlirValue *)NULL,
            .size = 0,
        }
    );
    EXPECT_NE(op.ptr, (void *)NULL);
    EXPECT_TRUE(mlirOperationVerify(op));
    EXPECT_TRUE(llzkOperationIsAApplyMapOp(op));
    mlirOperationDestroy(op);
    mlirOpBuilderDestroy(builder);
  }
}

TEST_F(CAPITest, llzk_apply_map_op_get_affine_map) {
  {
    auto builder = mlirOpBuilderCreate(ctx);
    auto location = mlirLocationUnknownGet(ctx);
    llvm::SmallVector<MlirAffineExpr> exprs({mlirAffineConstantExprGet(ctx, 1)});
    auto affine_map = mlirAffineMapGet(ctx, 0, 0, exprs.size(), exprs.data());
    auto op = llzkApplyMapOpBuildWithAffineMap(
        builder, location, affine_map,
        MlirValueRange {
            .values = (const MlirValue *)NULL,
            .size = 0,
        }
    );
    EXPECT_NE(op.ptr, (void *)NULL);
    EXPECT_TRUE(mlirOperationVerify(op));
    auto out_affine_map = llzkApplyMapOpGetAffineMap(op);
    EXPECT_TRUE(mlirAffineMapEqual(affine_map, out_affine_map));
    mlirOperationDestroy(op);
    mlirOpBuilderDestroy(builder);
  }
}

TEST_F(CAPITest, llzk_apply_map_op_get_dim_operands) {
  {
    auto builder = mlirOpBuilderCreate(ctx);
    auto location = mlirLocationUnknownGet(ctx);
    llvm::SmallVector<MlirAffineExpr> exprs({mlirAffineConstantExprGet(ctx, 1)});
    auto affine_map = mlirAffineMapGet(ctx, 0, 0, exprs.size(), exprs.data());
    auto op = llzkApplyMapOpBuildWithAffineMap(
        builder, location, affine_map,
        MlirValueRange {
            .values = (const MlirValue *)NULL,
            .size = 0,
        }
    );
    EXPECT_NE(op.ptr, (void *)NULL);
    EXPECT_TRUE(mlirOperationVerify(op));
    auto n_dims = llzkApplyMapOpGetNumDimOperands(op);
    llvm::SmallVector<MlirValue> dims(n_dims, MlirValue {.ptr = (void *)NULL});
    llzkApplyMapOpGetDimOperands(op, dims.data());
    EXPECT_EQ(dims.size(), 0);
    mlirOperationDestroy(op);
    mlirOpBuilderDestroy(builder);
  }
}

TEST_F(CAPITest, llzk_apply_map_op_get_symbol_operands) {
  {
    auto builder = mlirOpBuilderCreate(ctx);
    auto location = mlirLocationUnknownGet(ctx);
    llvm::SmallVector<MlirAffineExpr> exprs = {mlirAffineConstantExprGet(ctx, 1)};
    auto affine_map = mlirAffineMapGet(ctx, 0, 0, exprs.size(), exprs.data());
    auto op = llzkApplyMapOpBuildWithAffineMap(
        builder, location, affine_map,
        MlirValueRange {
            .values = (const MlirValue *)NULL,
            .size = 0,
        }
    );
    EXPECT_NE(op.ptr, (void *)NULL);
    EXPECT_TRUE(mlirOperationVerify(op));
    auto n_syms = llzkApplyMapOpGetNumSymbolOperands(op);
    llvm::SmallVector<MlirValue> syms(n_syms, {.ptr = (void *)NULL});
    llzkApplyMapOpGetSymbolOperands(op, syms.data());
    EXPECT_EQ(syms.size(), 0);
    mlirOperationDestroy(op);
    mlirOpBuilderDestroy(builder);
  }
}
