//===-- Builder.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Builder.h"

#include "CAPITestBase.h"

TEST_F(CAPITest, MlirOpBuilderCreate) {
  auto builder = mlirOpBuilderCreate(context);
  mlirOpBuilderDestroy(builder);
}
static void test_cb1(MlirOperation, void *) {}
static void test_cb2(MlirBlock, void *) {}

TEST_F(CAPITest, MlirOpBuilderCreateWithListener) {
  auto listener = mlirOpBuilderListenerCreate(test_cb1, test_cb2, NULL);
  auto builder = mlirOpBuilderCreateWithListener(context, listener);
  mlirOpBuilderDestroy(builder);
  mlirOpBuilderListenerDestroy(listener);
}

TEST_F(CAPITest, MlirOpBuilderListenerCreate) {
  auto listener = mlirOpBuilderListenerCreate(test_cb1, test_cb2, NULL);
  mlirOpBuilderListenerDestroy(listener);
}
