/**
 * The contents of this file are adapted from llvm/lib/Analysis/CallGraph.cpp
 */
#include "llzk/Dialect/LLZK/Analysis/AnalysisPasses.h"
#include "llzk/Dialect/LLZK/Analysis/ConstraintSummary.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>

namespace llzk {

#define GEN_PASS_DEF_CONSTRAINTSUMMARYPRINTERPASS
#include "llzk/Dialect/LLZK/Analysis/AnalysisPasses.h.inc"

class ConstraintSummaryPrinterPass
    : public impl::ConstraintSummaryPrinterPassBase<ConstraintSummaryPrinterPass> {
  llvm::raw_ostream &os;

public:
  explicit ConstraintSummaryPrinterPass(llvm::raw_ostream &ostream)
      : impl::ConstraintSummaryPrinterPassBase<ConstraintSummaryPrinterPass>(), os(ostream) {}

protected:
  void runOnOperation() override {
    markAllAnalysesPreserved();

    auto &cs = getAnalysis<ConstraintSummaryAnalysis>();
    for (auto &[_, summary] : cs) {
      summary.print(os);
    }
  }
};

std::unique_ptr<mlir::Pass>
createConstraintSummaryPrinterPass(llvm::raw_ostream &os = llvm::errs()) {
  return std::make_unique<ConstraintSummaryPrinterPass>(os);
}

} // namespace llzk