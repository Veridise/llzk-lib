//===-- r1cs-opt.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "r1cs/Dialect/IR/Dialect.h"
#include "r1cs/InitAllDialects.h"

#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  r1cs::registerAllDialects(registry);
  mlir::MLIRContext ctx;

  ctx.loadAllAvailableDialects();

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "R1CS Optimizer\n", registry));
}
