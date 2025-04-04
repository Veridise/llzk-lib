//===-- InitDialects.cpp - LLZK dialect registration ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/InitDialects.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h" // IWYU pragma: keep

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>

namespace llzk {
void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<llzk::LLZKDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
}
} // namespace llzk
