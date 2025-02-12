#include "llzk/Dialect/LLZK/Analysis/AnalysisPasses.h"
#include "llzk/Dialect/LLZK/Analysis/IntervalAnalysis.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>

namespace llzk {

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

    os << "ha!\n";
  }
};

std::unique_ptr<mlir::Pass>
createIntervalAnalysisPrinterPass(llvm::raw_ostream &os = llvm::errs()) {
  return std::make_unique<IntervalAnalysisPrinterPass>(os);
}

} // namespace llzk
