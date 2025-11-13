//===-- AnalysisUtil.h - Data-flow analysis utils ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Analysis/DataFlowFramework.h>

namespace llzk::dataflow {

/// LLZK: Added this utility to ensure analysis is performed for all structs
/// in a given module.
///
/// @brief Mark all operations from the top and included in the top operation
/// as live so the solver will perform dataflow analyses.
/// @param solver The solver.
/// @param top The top-level operation.
void markAllOpsAsLive(mlir::DataFlowSolver &solver, mlir::Operation *top);

} // namespace llzk::dataflow
