//===-- AnalysisPasses.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Pass/Pass.h>

#include <llvm/ADT/SCCIterator.h>
#include <llvm/ADT/STLExtras.h>

#include <cassert>
#include <map>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

namespace llzk {

std::unique_ptr<mlir::Pass> createCallGraphPrinterPass(llvm::raw_ostream &os);

std::unique_ptr<mlir::Pass> createCallGraphSCCsPrinterPass(llvm::raw_ostream &os);

std::unique_ptr<mlir::Pass> createConstraintDependencyGraphPrinterPass(llvm::raw_ostream &os);

std::unique_ptr<mlir::Pass> createIntervalAnalysisPrinterPass(llvm::raw_ostream &os);

#define GEN_PASS_REGISTRATION
#include "llzk/Dialect/LLZK/Analysis/AnalysisPasses.h.inc"

} // namespace llzk
