//===-- CAPITestBase.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk-c/InitDialects.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/IR.h>
#include <mlir-c/RegisterEverything.h>

#include <gtest/gtest.h>

class CAPITest : public ::testing::Test {
protected:
  MlirContext context;

  CAPITest() : context(mlirContextCreate()) {
    auto registry = mlirDialectRegistryCreate();
    mlirRegisterAllDialects(registry);
    llzkRegisterAllDialects(registry);
    mlirContextAppendDialectRegistry(context, registry);
    mlirContextLoadAllAvailableDialects(context);
    mlirDialectRegistryDestroy(registry);
  }

  ~CAPITest() override { mlirContextDestroy(context); }

  /// Helper to get IndexType
  inline MlirType createIndexType() const { return mlirIndexTypeGet(context); }

  /// Helper to create a simple IntegerAttr with IndexType
  inline MlirAttribute createIndexAttribute() const {
    return mlirIntegerAttrGet(createIndexType(), 0);
  }

  // Helper to create a simple test operation: `arith.constant 0 : index`
  MlirOperation createIndexOperation() {
    MlirType indexType = createIndexType();
    MlirOperationState op_state = mlirOperationStateGet(
        mlirStringRefCreateFromCString("arith.constant"), mlirLocationUnknownGet(context)
    );
    mlirOperationStateAddResults(&op_state, 1, &indexType);

    MlirNamedAttribute attr = mlirNamedAttributeGet(
        mlirIdentifierGet(context, mlirStringRefCreateFromCString("value")), createIndexAttribute()
    );
    mlirOperationStateAddAttributes(&op_state, 1, &attr);

    return mlirOperationCreate(&op_state);
  }
};
