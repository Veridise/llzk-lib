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

    if (!mlir::isa<mlir::ModuleOp>(getOperation())) {
      auto msg = "ConstraintSummaryPrinterPass error: should be run on ModuleOp!";
      getOperation()->emitError(msg);
      llvm::report_fatal_error(msg);
    }

    auto &cs = getAnalysis<ConstraintSummaryModuleAnalysis>();
    for (auto &[structDef, summary_ptr] : cs) {
      os << "Constraint Summary for " << const_cast<StructDefOp &>(structDef).getName() << ":\n";
      summary_ptr->print(os);
    }
  }
};

std::unique_ptr<mlir::Pass>
createConstraintSummaryPrinterPass(llvm::raw_ostream &os = llvm::errs()) {
  return std::make_unique<ConstraintSummaryPrinterPass>(os);
}

} // namespace llzk