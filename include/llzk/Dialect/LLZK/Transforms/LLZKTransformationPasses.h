//===-- LLZKTransformationPasses.h ------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Pass/Pass.h>

namespace llzk {

std::unique_ptr<mlir::Pass> createInlineIncludesPass();
std::unique_ptr<mlir::Pass> createFlatteningPass();

std::unique_ptr<mlir::Pass> createRedundantReadAndWriteEliminationPass();

std::unique_ptr<mlir::Pass> createRedundantOperationEliminationPass();

std::unique_ptr<mlir::Pass> createUnusedDeclarationEliminationPass();

void registerTransformationPassPipelines();

#define GEN_PASS_REGISTRATION
#include "llzk/Dialect/LLZK/Transforms/LLZKTransformationPasses.h.inc"

}; // namespace llzk
