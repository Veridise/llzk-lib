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

#include <mlir-c/IR.h>
#include <mlir-c/RegisterEverything.h>

#include <gtest/gtest.h>

class CAPITest : public ::testing::Test {
protected:
  MlirContext ctx;

  CAPITest() : ctx(mlirContextCreate()) {
    auto registry = mlirDialectRegistryCreate();
    mlirRegisterAllDialects(registry);
    llzkRegisterAllDialects(registry);
    mlirContextAppendDialectRegistry(ctx, registry);
    mlirContextLoadAllAvailableDialects(ctx);
    mlirDialectRegistryDestroy(registry);
  }

  ~CAPITest() override { mlirContextDestroy(ctx); }
};
