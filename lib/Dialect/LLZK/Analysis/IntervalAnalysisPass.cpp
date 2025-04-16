//===-- IntervalAnalysisPass.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-print-interval-analysis` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/Analysis/AnalysisPasses.h"
#include "llzk/Dialect/LLZK/Analysis/IntervalAnalysis.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Types.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>

namespace llzk {

#define GEN_PASS_DECL_INTERVALANALYSISPRINTERPASS
#define GEN_PASS_DEF_INTERVALANALYSISPRINTERPASS
#include "llzk/Dialect/LLZK/Analysis/AnalysisPasses.h.inc"

class IntervalAnalysisPrinterPass
    : public impl::IntervalAnalysisPrinterPassBase<IntervalAnalysisPrinterPass> {
  llvm::raw_ostream &os;

public:
  explicit IntervalAnalysisPrinterPass(llvm::raw_ostream &ostream)
      : impl::IntervalAnalysisPrinterPassBase<IntervalAnalysisPrinterPass>(), os(ostream) {}

protected:
  void runOnOperation() override {
    markAllAnalysesPreserved();

    if (!mlir::isa<mlir::ModuleOp>(getOperation())) {
      auto msg = "IntervalAnalysisPrinterPass error: should be run on ModuleOp!";
      getOperation()->emitError(msg);
      llvm::report_fatal_error(msg);
    }

    auto &mia = getAnalysis<ModuleIntervalAnalysis>();
    mia.setField(Field::getField(fieldName.c_str()));
    auto am = getAnalysisManager();
    mia.runAnalysis(am);

    for (auto &[s, si] : mia) {
      auto &structDef = const_cast<StructDefOp &>(s);
      // Don't print the analysis for built-ins.
      if (isSignalType(structDef.getType())) {
        continue;
      }
      auto fullName = getPathFromTopRoot(structDef);
      ensure(
          mlir::succeeded(fullName),
          "could not resolve fully qualified name of struct " + mlir::Twine(structDef.getName())
      );
      os << fullName.value() << ' ';
      si.get().print(os, printSolverConstraints);
    }
  }
};

std::unique_ptr<mlir::Pass>
createIntervalAnalysisPrinterPass(llvm::raw_ostream &os = llvm::errs()) {
  return std::make_unique<IntervalAnalysisPrinterPass>(os);
}

} // namespace llzk
