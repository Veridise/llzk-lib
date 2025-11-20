//===-- AnalysisUtil.cpp - Data-flow analysis utils -------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/AnalysisUtil.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>

using namespace mlir;

using Executable = mlir::dataflow::Executable;

namespace llzk::dataflow {

void markAllOpsAsLive(DataFlowSolver &solver, Operation *top) {
  for (Region &region : top->getRegions()) {
    for (Block &block : region) {
      ProgramPoint *point = solver.getProgramPointBefore(&block);
      (void)solver.getOrCreateState<Executable>(point)->setToLive();
      for (Operation &oper : block) {
        markAllOpsAsLive(solver, &oper);
      }
    }
  }
}

} // namespace llzk::dataflow
