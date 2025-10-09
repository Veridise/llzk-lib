//===-- Struct.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Struct.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>

#include <llvm/ADT/SmallVector.h>

#include <gtest/gtest.h>

#include "../CAPITestBase.h"

TEST_F(CAPITest, mlir_get_dialect_handle_llzk_component) {
  {
    mlirGetDialectHandle__llzk__component__();
  }
}

TEST_F(CAPITest, llzk_struct_type_get) {
  {
    auto s = mlirStringRefCreateFromCString("T");
    auto sym = mlirFlatSymbolRefAttrGet(context, s);
    auto t = llzkStructTypeGet(sym);
    EXPECT_NE(t.ptr, (void *)NULL);
  }
}

TEST_F(CAPITest, llzk_struct_type_get_with_array_attr) {
  {
    auto s = mlirStringRefCreateFromCString("T");
    auto sym = mlirFlatSymbolRefAttrGet(context, s);
    llvm::SmallVector<MlirAttribute> attrs(
        {mlirFlatSymbolRefAttrGet(context, mlirStringRefCreateFromCString("A"))}
    );
    auto a = mlirArrayAttrGet(context, attrs.size(), attrs.data());
    auto t = llzkStructTypeGetWithArrayAttr(sym, a);
    EXPECT_NE(t.ptr, (void *)NULL);
  }
}

TEST_F(CAPITest, llzk_struct_type_get_with_attrs) {
  {
    auto s = mlirStringRefCreateFromCString("T");
    auto sym = mlirFlatSymbolRefAttrGet(context, s);
    llvm::SmallVector<MlirAttribute> attrs(
        {mlirFlatSymbolRefAttrGet(context, mlirStringRefCreateFromCString("A"))}
    );
    auto t = llzkStructTypeGetWithAttrs(sym, attrs.size(), attrs.data());
    EXPECT_NE(t.ptr, (void *)NULL);
  }
}

TEST_F(CAPITest, llzk_type_is_a_struct_type) {
  {
    auto s = mlirStringRefCreateFromCString("T");
    auto sym = mlirFlatSymbolRefAttrGet(context, s);
    auto t = llzkStructTypeGet(sym);
    EXPECT_NE(t.ptr, (void *)NULL);
    EXPECT_TRUE(llzkTypeIsAStructType(t));
  }
}

TEST_F(CAPITest, llzk_struct_type_get_name) {
  {
    auto s = mlirStringRefCreateFromCString("T");
    auto sym = mlirFlatSymbolRefAttrGet(context, s);
    auto t = llzkStructTypeGet(sym);
    EXPECT_NE(t.ptr, (void *)NULL);
    EXPECT_TRUE(mlirAttributeEqual(sym, llzkStructTypeGetName(t)));
  }
}

TEST_F(CAPITest, llzk_struct_type_get_params) {
  {
    auto s = mlirStringRefCreateFromCString("T");
    auto sym = mlirFlatSymbolRefAttrGet(context, s);
    llvm::SmallVector<MlirAttribute> attrs(
        {mlirFlatSymbolRefAttrGet(context, mlirStringRefCreateFromCString("A"))}
    );
    auto a = mlirArrayAttrGet(context, attrs.size(), attrs.data());
    auto t = llzkStructTypeGetWithArrayAttr(sym, a);
    EXPECT_NE(t.ptr, (void *)NULL);
    EXPECT_TRUE(mlirAttributeEqual(a, llzkStructTypeGetParams(t)));
  }
}

struct TestOp {
  MlirOperation op;

  ~TestOp() { mlirOperationDestroy(op); }
};

class StructDefTest : public CAPITest {

protected:
  MlirOperation new_struct() const {
    {
      auto struct_name = mlirFlatSymbolRefAttrGet(context, mlirStringRefCreateFromCString("S"));
      auto name = mlirStringRefCreateFromCString("struct.new");
      auto location = mlirLocationUnknownGet(context);
      auto result = llzkStructTypeGet(struct_name);
      auto op_state = mlirOperationStateGet(name, location);
      mlirOperationStateAddResults(&op_state, 1, &result);
      return mlirOperationCreate(&op_state);
    }
  }
  TestOp test_op() const {
    {
      auto elt_type = mlirIndexTypeGet(context);
      auto name = mlirStringRefCreateFromCString("arith.constant");
      auto attr_name = mlirIdentifierGet(context, mlirStringRefCreateFromCString("value"));
      auto location = mlirLocationUnknownGet(context);
      llvm::SmallVector<MlirType> results({elt_type});
      auto attr = mlirIntegerAttrGet(elt_type, 1);
      llvm::SmallVector<MlirNamedAttribute> attrs({mlirNamedAttributeGet(attr_name, attr)});
      auto op_state = mlirOperationStateGet(name, location);
      mlirOperationStateAddResults(&op_state, results.size(), results.data());
      mlirOperationStateAddAttributes(&op_state, attrs.size(), attrs.data());
      return {
          .op = mlirOperationCreate(&op_state),
      };
    }
  }
};

TEST_F(StructDefTest, llzk_operation_is_a_struct_def_op) {
  auto op = test_op();
  {
    EXPECT_TRUE(!llzkOperationIsAStructDefOp(op.op));
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_body) {
  auto op = test_op();
  {
    if (llzkOperationIsAStructDefOp(op.op)) {
      llzkStructDefOpGetBody(op.op);
    }
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_body_region) {
  auto op = test_op();
  {
    if (llzkOperationIsAStructDefOp(op.op)) {
      llzkStructDefOpGetBodyRegion(op.op);
    }
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_type) {
  auto op = test_op();
  {
    if (llzkOperationIsAStructDefOp(op.op)) {
      llzkStructDefOpGetType(op.op);
    }
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_type_with_params) {
  auto op = test_op();
  {
    if (llzkOperationIsAStructDefOp(op.op)) {
      auto attrs = mlirArrayAttrGet(mlirOperationGetContext(op.op), 0, (const MlirAttribute *)NULL);
      llzkStructDefOpGetTypeWithParams(op.op, attrs);
    }
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_field_def) {
  auto op = test_op();
  {
    if (llzkOperationIsAStructDefOp(op.op)) {
      auto name = mlirStringRefCreateFromCString("p");
      llzkStructDefOpGetFieldDef(op.op, name);
    }
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_field_defs) {
  auto op = test_op();
  {
    if (llzkOperationIsAStructDefOp(op.op)) {
      llzkStructDefOpGetFieldDefs(op.op, (MlirOperation *)NULL);
    }
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_num_field_defs) {
  auto op = test_op();
  {
    if (llzkOperationIsAStructDefOp(op.op)) {
      llzkStructDefOpGetNumFieldDefs(op.op);
    }
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_has_columns) {
  auto op = test_op();
  {
    if (llzkOperationIsAStructDefOp(op.op)) {
      llzkStructDefOpGetHasColumns(op.op);
    }
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_compute_func_op) {
  auto op = test_op();
  {
    if (llzkOperationIsAStructDefOp(op.op)) {
      llzkStructDefOpGetComputeFuncOp(op.op);
    }
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_constrain_func_op) {
  auto op = test_op();
  {
    if (llzkOperationIsAStructDefOp(op.op)) {
      llzkStructDefOpGetConstrainFuncOp(op.op);
    }
  }
}

static char *cmalloc(size_t s) { return (char *)malloc(s); }

TEST_F(StructDefTest, llzk_struct_def_op_get_header_string) {
  auto op = test_op();
  {
    if (llzkOperationIsAStructDefOp(op.op)) {
      intptr_t size = 0;
      auto str = llzkStructDefOpGetHeaderString(op.op, &size, cmalloc);
      free(static_cast<void *>(const_cast<char *>(str)));
    }
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_has_param_name) {
  auto op = test_op();
  {
    if (llzkOperationIsAStructDefOp(op.op)) {
      auto name = mlirStringRefCreateFromCString("p");
      llzkStructDefOpGetHasParamName(op.op, name);
    }
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_fully_qualified_name) {
  auto op = test_op();
  {
    if (llzkOperationIsAStructDefOp(op.op)) {
      llzkStructDefOpGetFullyQualifiedName(op.op);
    }
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_is_main_component) {
  auto op = test_op();
  {
    if (llzkOperationIsAStructDefOp(op.op)) {
      llzkStructDefOpGetIsMainComponent(op.op);
    }
  }
}

TEST_F(StructDefTest, llzk_operation_is_a_field_def_op) {
  auto op = test_op();
  {
    EXPECT_TRUE(!llzkOperationIsAFieldDefOp(op.op));
  }
}

TEST_F(StructDefTest, llzk_field_def_op_get_has_public_attr) {
  auto op = test_op();
  {
    if (llzkOperationIsAFieldDefOp(op.op)) {
      llzkFieldDefOpGetHasPublicAttr(op.op);
    }
  }
}

TEST_F(StructDefTest, llzk_field_def_op_set_public_attr) {
  auto op = test_op();
  {
    if (llzkOperationIsAFieldDefOp(op.op)) {
      llzkFieldDefOpSetPublicAttr(op.op, true);
    }
  }
}

TEST_F(StructDefTest, llzk_field_read_op_build) {
  {
    auto builder = mlirOpBuilderCreate(context);
    auto location = mlirLocationUnknownGet(context);
    auto index_type = mlirIndexTypeGet(context);
    auto _struct = new_struct();
    auto struct_value = mlirOperationGetResult(_struct, 0);
    auto op = llzkFieldReadOpBuild(
        builder, location, index_type, struct_value, mlirStringRefCreateFromCString("f")
    );

    mlirOperationDestroy(op);
    mlirOperationDestroy(_struct);
    mlirOpBuilderDestroy(builder);
  }
}

TEST_F(StructDefTest, llzk_field_read_op_build_with_affine_map_distance) {
  {
    auto builder = mlirOpBuilderCreate(context);
    auto location = mlirLocationUnknownGet(context);
    auto index_type = mlirIndexTypeGet(context);
    auto _struct = new_struct();
    auto struct_value = mlirOperationGetResult(_struct, 0);

    llvm::SmallVector<MlirAffineExpr> exprs({mlirAffineConstantExprGet(context, 1)});
    auto affine_map = mlirAffineMapGet(context, 0, 0, exprs.size(), exprs.data());
    auto op = llzkFieldReadOpBuildWithAffineMapDistance(
        builder, location, index_type, struct_value, mlirStringRefCreateFromCString("f"),
        affine_map,
        MlirValueRange {
            .values = (const MlirValue *)NULL,
            .size = 0,
        },
        0
    );

    mlirOperationDestroy(op);
    mlirOperationDestroy(_struct);
    mlirOpBuilderDestroy(builder);
  }
}

TEST_F(StructDefTest, llzk_field_read_op_builder_with_const_param_distance) {
  {
    auto builder = mlirOpBuilderCreate(context);
    auto location = mlirLocationUnknownGet(context);
    auto index_type = mlirIndexTypeGet(context);
    auto _struct = new_struct();
    auto struct_value = mlirOperationGetResult(_struct, 0);

    auto op = llzkFieldReadOpBuildWithConstParamDistance(
        builder, location, index_type, struct_value, mlirStringRefCreateFromCString("f"),
        mlirStringRefCreateFromCString("N")
    );

    mlirOperationDestroy(op);
    mlirOperationDestroy(_struct);
    mlirOpBuilderDestroy(builder);
  }
}

TEST_F(StructDefTest, llzk_field_read_op_build_with_literal_distance) {
  {
    auto builder = mlirOpBuilderCreate(context);
    auto location = mlirLocationUnknownGet(context);
    auto index_type = mlirIndexTypeGet(context);
    auto _struct = new_struct();
    auto struct_value = mlirOperationGetResult(_struct, 0);

    auto op = llzkFieldReadOpBuildWithLiteralDistance(
        builder, location, index_type, struct_value, mlirStringRefCreateFromCString("f"), 1
    );

    mlirOperationDestroy(op);
    mlirOperationDestroy(_struct);
    mlirOpBuilderDestroy(builder);
  }
}
