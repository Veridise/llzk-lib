//===-- Transforms.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Transforms.h"

#include "CAPITestBase.h"

TEST_F(CAPITest, RegisterTransformationPassesAndCreate) {
  mlirRegisterLLZKTransformationPasses();
  auto manager = mlirPassManagerCreate(context);

  auto pass1 = mlirCreateLLZKTransformationRedundantOperationEliminationPass();
  auto pass2 = mlirCreateLLZKTransformationRedundantReadAndWriteEliminationPass();
  auto pass3 = mlirCreateLLZKTransformationUnusedDeclarationEliminationPass();
  mlirPassManagerAddOwnedPass(manager, pass1);
  mlirPassManagerAddOwnedPass(manager, pass2);
  mlirPassManagerAddOwnedPass(manager, pass3);

  mlirPassManagerDestroy(manager);
}

TEST_F(CAPITest, RegisterRedundantOperationEliminationPassAndCreate) {
  mlirRegisterLLZKTransformationRedundantOperationEliminationPass();
  auto manager = mlirPassManagerCreate(context);

  auto pass = mlirCreateLLZKTransformationRedundantOperationEliminationPass();
  mlirPassManagerAddOwnedPass(manager, pass);

  mlirPassManagerDestroy(manager);
}

TEST_F(CAPITest, RegisterRedudantReadAndWriteEliminationPassAndCreate) {
  mlirRegisterLLZKTransformationRedundantReadAndWriteEliminationPass();
  auto manager = mlirPassManagerCreate(context);

  auto pass = mlirCreateLLZKTransformationRedundantReadAndWriteEliminationPass();
  mlirPassManagerAddOwnedPass(manager, pass);

  mlirPassManagerDestroy(manager);
}

TEST_F(CAPITest, RegisterUnuusedDeclarationEliminationPassAndCreate) {
  mlirRegisterLLZKTransformationUnusedDeclarationEliminationPass();
  auto manager = mlirPassManagerCreate(context);

  auto pass = mlirCreateLLZKTransformationUnusedDeclarationEliminationPass();
  mlirPassManagerAddOwnedPass(manager, pass);

  mlirPassManagerDestroy(manager);
}
