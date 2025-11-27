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

// Include the auto-generated tests
#include "llzk/Dialect/Struct/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Struct/IR/Ops.capi.test.cpp.inc"
#include "llzk/Dialect/Struct/IR/Types.capi.test.cpp.inc"

TEST_F(CAPITest, llzk_struct_type_get) {
  auto s = mlirStringRefCreateFromCString("T");
  auto sym = mlirFlatSymbolRefAttrGet(context, s);
  auto t = llzkStructStructTypeGet(sym);
  EXPECT_NE(t.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_struct_type_get_with_array_attr) {
  auto s = mlirStringRefCreateFromCString("T");
  auto sym = mlirFlatSymbolRefAttrGet(context, s);
  llvm::SmallVector<MlirAttribute> attrs(
      {mlirFlatSymbolRefAttrGet(context, mlirStringRefCreateFromCString("A"))}
  );
  auto a = mlirArrayAttrGet(context, attrs.size(), attrs.data());
  auto t = llzkStructStructTypeGetWithArrayAttr(sym, a);
  EXPECT_NE(t.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_struct_type_get_with_attrs) {
  auto s = mlirStringRefCreateFromCString("T");
  auto sym = mlirFlatSymbolRefAttrGet(context, s);
  llvm::SmallVector<MlirAttribute> attrs(
      {mlirFlatSymbolRefAttrGet(context, mlirStringRefCreateFromCString("A"))}
  );
  auto t = llzkStructStructTypeGetWithAttrs(sym, attrs.size(), attrs.data());
  EXPECT_NE(t.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_type_is_a_struct_type_pass) {
  auto s = mlirStringRefCreateFromCString("T");
  auto sym = mlirFlatSymbolRefAttrGet(context, s);
  auto t = llzkStructStructTypeGet(sym);
  EXPECT_NE(t.ptr, (void *)NULL);
  EXPECT_TRUE(llzkTypeIsAStructStructType(t));
}

TEST_F(CAPITest, llzk_struct_type_get_name) {
  auto s = mlirStringRefCreateFromCString("T");
  auto sym = mlirFlatSymbolRefAttrGet(context, s);
  auto t = llzkStructStructTypeGet(sym);
  EXPECT_NE(t.ptr, (void *)NULL);
  EXPECT_TRUE(mlirAttributeEqual(sym, llzkStructStructTypeGetNameRef(t)));
}

TEST_F(CAPITest, llzk_struct_type_get_params) {
  auto s = mlirStringRefCreateFromCString("T");
  auto sym = mlirFlatSymbolRefAttrGet(context, s);
  llvm::SmallVector<MlirAttribute> attrs(
      {mlirFlatSymbolRefAttrGet(context, mlirStringRefCreateFromCString("A"))}
  );
  auto a = mlirArrayAttrGet(context, attrs.size(), attrs.data());
  auto t = llzkStructStructTypeGetWithArrayAttr(sym, a);
  EXPECT_NE(t.ptr, (void *)NULL);
  EXPECT_TRUE(mlirAttributeEqual(a, llzkStructStructTypeGetParams(t)));
}

struct TestOp {
  MlirOperation op;

  ~TestOp() { mlirOperationDestroy(op); }
};

struct StructDefTest : public CAPITest {
  MlirOperation make_struct_def_op() const {
    auto name = mlirStringRefCreateFromCString("struct.def");
    auto location = mlirLocationUnknownGet(context);
    llvm::SmallVector<MlirNamedAttribute> attrs({mlirNamedAttributeGet(
        mlirIdentifierGet(context, mlirStringRefCreateFromCString("sym_name")),
        mlirStringAttrGet(context, mlirStringRefCreateFromCString("S"))
    )});
    auto op_state = mlirOperationStateGet(name, location);
    mlirOperationStateAddAttributes(&op_state, attrs.size(), attrs.data());
    return mlirOperationCreate(&op_state);
  }

  MlirOperation make_struct_new_op() const {
    auto struct_name = mlirFlatSymbolRefAttrGet(context, mlirStringRefCreateFromCString("S"));
    auto name = mlirStringRefCreateFromCString("struct.new");
    auto location = mlirLocationUnknownGet(context);
    auto result = llzkStructStructTypeGet(struct_name);
    auto op_state = mlirOperationStateGet(name, location);
    mlirOperationStateAddResults(&op_state, 1, &result);
    return mlirOperationCreate(&op_state);
  }

  MlirOperation make_field_def_op() const {
    auto name = mlirStringRefCreateFromCString("struct.field");
    auto location = mlirLocationUnknownGet(context);
    llvm::SmallVector<MlirNamedAttribute> attrs(
        {mlirNamedAttributeGet(
             mlirIdentifierGet(context, mlirStringRefCreateFromCString("sym_name")),
             mlirStringAttrGet(context, mlirStringRefCreateFromCString("S"))
         ),
         mlirNamedAttributeGet(
             mlirIdentifierGet(context, mlirStringRefCreateFromCString("type")),
             mlirTypeAttrGet(createIndexType())
         )}
    );
    auto op_state = mlirOperationStateGet(name, location);
    mlirOperationStateAddAttributes(&op_state, attrs.size(), attrs.data());
    return mlirOperationCreate(&op_state);
  }

  TestOp test_op() const {
    auto elt_type = createIndexType();
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
};

TEST_F(StructDefTest, llzk_operation_is_a_struct_def_op_pass) {
  auto op = make_struct_def_op();
  EXPECT_TRUE(llzkOperationIsAStructStructDefOp(op));
}

TEST_F(StructDefTest, llzk_struct_def_op_get_body) {
  auto op = test_op();
  if (llzkOperationIsAStructStructDefOp(op.op)) {
    llzkStructStructDefOpGetBody(op.op);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_body_region) {
  auto op = test_op();
  if (llzkOperationIsAStructStructDefOp(op.op)) {
    llzkStructStructDefOpGetBodyRegion(op.op);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_type) {
  auto op = test_op();
  if (llzkOperationIsAStructStructDefOp(op.op)) {
    llzkStructStructDefOpGetType(op.op);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_type_with_params) {
  auto op = test_op();
  if (llzkOperationIsAStructStructDefOp(op.op)) {
    auto attrs = mlirArrayAttrGet(mlirOperationGetContext(op.op), 0, (const MlirAttribute *)NULL);
    llzkStructStructDefOpGetTypeWithParams(op.op, attrs);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_field_def) {
  auto op = test_op();
  if (llzkOperationIsAStructStructDefOp(op.op)) {
    MlirIdentifier name =
        mlirIdentifierGet(mlirOperationGetContext(op.op), mlirStringRefCreateFromCString("p"));
    llzkStructStructDefOpGetFieldDef(op.op, name);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_field_defs) {
  auto op = test_op();
  if (llzkOperationIsAStructStructDefOp(op.op)) {
    llzkStructStructDefOpGetFieldDefs(op.op, (MlirOperation *)NULL);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_num_field_defs) {
  auto op = test_op();
  if (llzkOperationIsAStructStructDefOp(op.op)) {
    llzkStructStructDefOpGetNumFieldDefs(op.op);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_has_columns) {
  auto op = test_op();
  if (llzkOperationIsAStructStructDefOp(op.op)) {
    llzkStructStructDefOpHasColumns(op.op);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_compute_func_op) {
  auto op = test_op();
  if (llzkOperationIsAStructStructDefOp(op.op)) {
    llzkStructStructDefOpGetComputeFuncOp(op.op);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_constrain_func_op) {
  auto op = test_op();
  if (llzkOperationIsAStructStructDefOp(op.op)) {
    llzkStructStructDefOpGetComputeFuncOp(op.op);
  }
}

static char *cmalloc(size_t s) { return (char *)malloc(s); }

TEST_F(StructDefTest, llzk_struct_def_op_get_header_string) {
  auto op = test_op();
  if (llzkOperationIsAStructStructDefOp(op.op)) {
    intptr_t size = 0;
    auto str = llzkStructStructDefOpGetHeaderString(op.op, &size, cmalloc);
    free(static_cast<void *>(const_cast<char *>(str)));
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_has_param_name) {
  auto op = test_op();
  if (llzkOperationIsAStructStructDefOp(op.op)) {
    auto name = mlirStringRefCreateFromCString("p");
    llzkStructStructDefOpGetHasParamName(op.op, name);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_fully_qualified_name) {
  auto op = test_op();
  if (llzkOperationIsAStructStructDefOp(op.op)) {
    llzkStructStructDefOpGetFullyQualifiedName(op.op);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_is_main_component) {
  auto op = test_op();
  if (llzkOperationIsAStructStructDefOp(op.op)) {
    llzkStructStructDefOpIsMainComponent(op.op);
  }
}

TEST_F(StructDefTest, llzk_operation_is_a_field_def_op_pass) {
  auto op = make_field_def_op();
  EXPECT_TRUE(llzkOperationIsAStructFieldDefOp(op));
}

TEST_F(StructDefTest, llzk_field_def_op_has_public_attr) {
  auto op = test_op();
  if (llzkOperationIsAStructFieldDefOp(op.op)) {
    llzkStructFieldDefOpHasPublicAttr(op.op);
  }
}

TEST_F(StructDefTest, llzk_field_def_op_set_public_attr) {
  auto op = test_op();
  if (llzkOperationIsAStructFieldDefOp(op.op)) {
    llzkStructFieldDefOpSetPublicAttr(op.op, true);
  }
}

struct FieldReadOpBuildFuncHelper : public TestAnyBuildFuncHelper<StructDefTest> {
  MlirOperation struct_new_op;
  bool callIsA(MlirOperation op) override { return llzkOperationIsAStructFieldReadOp(op); }
  ~FieldReadOpBuildFuncHelper() override { mlirOperationDestroy(this->struct_new_op); }
};

TEST_F(StructDefTest, llzk_field_read_op_build) {
  struct : FieldReadOpBuildFuncHelper {
    MlirOperation callBuild(
        const StructDefTest &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      this->struct_new_op = testClass.make_struct_new_op();
      auto index_type = testClass.createIndexType();
      auto struct_value = mlirOperationGetResult(struct_new_op, 0);
      return llzkStructFieldReadOpBuild(
          builder, location, index_type, struct_value, mlirStringRefCreateFromCString("f")
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(StructDefTest, llzk_field_read_op_build_with_affine_map_distance) {
  struct : FieldReadOpBuildFuncHelper {
    MlirOperation callBuild(
        const StructDefTest &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      this->struct_new_op = testClass.make_struct_new_op();
      auto index_type = testClass.createIndexType();
      auto struct_value = mlirOperationGetResult(struct_new_op, 0);
      llvm::SmallVector<MlirAffineExpr> exprs({mlirAffineConstantExprGet(testClass.context, 1)});
      auto affine_map = mlirAffineMapGet(testClass.context, 0, 0, exprs.size(), exprs.data());
      return llzkStructFieldReadOpBuildWithAffineMapDistance(
          builder, location, index_type, struct_value, mlirStringRefCreateFromCString("f"),
          affine_map,
          MlirValueRange {
              .values = (const MlirValue *)NULL,
              .size = 0,
          },
          0
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(StructDefTest, llzk_field_read_op_builder_with_const_param_distance) {
  struct : FieldReadOpBuildFuncHelper {
    MlirOperation callBuild(
        const StructDefTest &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      this->struct_new_op = testClass.make_struct_new_op();
      auto index_type = testClass.createIndexType();
      auto struct_value = mlirOperationGetResult(struct_new_op, 0);
      return llzkStructFieldReadOpBuildWithConstParamDistance(
          builder, location, index_type, struct_value, mlirStringRefCreateFromCString("f"),
          mlirStringRefCreateFromCString("N")
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(StructDefTest, llzk_field_read_op_build_with_literal_distance) {
  struct : FieldReadOpBuildFuncHelper {
    MlirOperation callBuild(
        const StructDefTest &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      this->struct_new_op = testClass.make_struct_new_op();
      auto index_type = testClass.createIndexType();
      auto struct_value = mlirOperationGetResult(struct_new_op, 0);
      return llzkStructFieldReadOpBuildWithLiteralDistance(
          builder, location, index_type, struct_value, mlirStringRefCreateFromCString("f"), 1
      );
    }
  } helper;
  helper.run(*this);
}
