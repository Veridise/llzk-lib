/**
 * The contents of this file are adapted from llvm/lib/Analysis/CallGraph.cpp
 */
#include "llzk/Dialect/LLZK/Analysis/AnalysisPasses.h"
#include "llzk/Dialect/LLZK/Analysis/ConstraintSummary.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

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
    for (auto &[s, summary_ptr] : cs) {
      auto &structDef = const_cast<StructDefOp &>(s);
      auto fullName = getPathFromRoot(structDef);
      debug::ensure(
          mlir::succeeded(fullName),
          "could not resolve fully qualified name of struct " + mlir::Twine(structDef.getName())
      );
      os << fullName.value() << ' ';
      summary_ptr->print(os);
    }
  }
};

std::unique_ptr<mlir::Pass>
createConstraintSummaryPrinterPass(llvm::raw_ostream &os = llvm::errs()) {
  return std::make_unique<ConstraintSummaryPrinterPass>(os);
}

} // namespace llzk
