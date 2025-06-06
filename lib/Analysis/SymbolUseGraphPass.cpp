//===-- SymbolUseGraphPass.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-print-symbol-use-graph` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/AnalysisPasses.h"
#include "llzk/Analysis/SymbolUseGraph.h"

namespace llzk {

#define GEN_PASS_DECL_SYMBOLUSEGRAPHPRINTERPASS
#define GEN_PASS_DEF_SYMBOLUSEGRAPHPRINTERPASS
#include "llzk/Analysis/AnalysisPasses.h.inc"

using namespace component;

class SymbolUseGraphPass : public impl::SymbolUseGraphPrinterPassBase<SymbolUseGraphPass> {
public:
  SymbolUseGraphPass() : impl::SymbolUseGraphPrinterPassBase<SymbolUseGraphPass>() {}

protected:
  void runOnOperation() override {
    markAllAnalysesPreserved();

    SymbolUseGraph &a = getAnalysis<SymbolUseGraph>();
    if (saveDotGraph) {
      a.dumpToDotFile();
    }
    a.print(toStream(outputStream));
  }
};

std::unique_ptr<mlir::Pass> createSymbolUseGraphPrinterPass() {
  return std::make_unique<SymbolUseGraphPass>();
}

} // namespace llzk
