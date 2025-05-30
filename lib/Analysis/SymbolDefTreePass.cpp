//===-- SymbolDefTreePass.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-print-symbol-def-tree` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/AnalysisPasses.h"
#include "llzk/Analysis/SymbolDefTree.h"

namespace llzk {

#define GEN_PASS_DECL_SYMBOLDEFTREEPRINTERPASS
#define GEN_PASS_DEF_SYMBOLDEFTREEPRINTERPASS
#include "llzk/Analysis/AnalysisPasses.h.inc"

using namespace component;

class SymbolDefTreePass : public impl::SymbolDefTreePrinterPassBase<SymbolDefTreePass> {
public:
  SymbolDefTreePass() : impl::SymbolDefTreePrinterPassBase<SymbolDefTreePass>() {}

protected:
  void runOnOperation() override {
    markAllAnalysesPreserved();
    getAnalysis<SymbolDefTree>().print(toStream(outputStream));
  }
};

std::unique_ptr<mlir::Pass> createSymbolDefTreePrinterPass() {
  return std::make_unique<SymbolDefTreePass>();
}

} // namespace llzk
