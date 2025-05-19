//===-- TransformationPassPipeline.cpp --------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements logic for registering the `-llzk-remove-unnecessary-ops`
/// and `-llzk-remove-unnecessary-ops-and-defs` pipelines.
///
//===----------------------------------------------------------------------===//

#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>

using namespace mlir;

namespace llzk {

struct FullPolyLoweringOptions : public mlir::PassPipelineOptions<FullPolyLoweringOptions> {
  Option<unsigned> maxDegree {
      *this, "max-degree", llvm::cl::desc("Maximum polynomial degree (must be ≥ 2)"),
      llvm::cl::init(2)
  };
};

void addRemoveUnnecessaryOpsAndDefsPipeline(mlir::OpPassManager &pm) {
  pm.addPass(llzk::createRedundantReadAndWriteEliminationPass());
  pm.addPass(llzk::createRedundantOperationEliminationPass());
  pm.addPass(llzk::createUnusedDeclarationEliminationPass());
}

void registerTransformationPassPipelines() {
  PassPipelineRegistration<>(
      "llzk-remove-unnecessary-ops",
      "Remove unnecessary operations, such as redundant reads or repeated constraints",
      [](OpPassManager &pm) {
    pm.addPass(createRedundantReadAndWriteEliminationPass());
    pm.addPass(createRedundantOperationEliminationPass());
  }
  );

  PassPipelineRegistration<>(
      "llzk-remove-unnecessary-ops-and-defs",
      "Remove unnecessary operations, field definitions, and struct definitions",
      [](OpPassManager &pm) { addRemoveUnnecessaryOpsAndDefsPipeline(pm); }
  );

  mlir::PassPipelineRegistration<FullPolyLoweringOptions>(
      "llzk-full-poly-lowering",
      "Lower all polynomial constraints to a given max degree, then remove unnecessary operations "
      "and definitions.",
      [](mlir::OpPassManager &pm, const FullPolyLoweringOptions &opts) {
    if (opts.maxDegree < 2) {
      llvm::errs() << "llzk-full-poly-lowering: max-degree must be >= 2\n";
      exit(1); // or handle more gracefully
    }

    // 1. Degree lowering
    auto polyPass = llzk::createPolyLoweringPass(opts.maxDegree);
    pm.addPass(std::move(polyPass));

    // 2. Cleanup
    addRemoveUnnecessaryOpsAndDefsPipeline(pm);
  }
  );
}

} // namespace llzk
