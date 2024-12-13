/**
 * The contents of this file are adapted from llvm/lib/Analysis/CallGraph.cpp
 */
#include "llzk/Dialect/LLZK/Analysis/CallGraphPasses.h"
#include "llzk/Dialect/LLZK/Analysis/CallGraphAnalyses.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>

namespace llzk {

#define GEN_PASS_DEF_CALLGRAPHPRINTERPASS
#define GEN_PASS_DEF_CALLGRAPHSCCSPRINTERPASS
#include "llzk/Dialect/LLZK/Analysis/CallGraphPasses.h.inc"

class CallGraphPrinterPass : public impl::CallGraphPrinterPassBase<CallGraphPrinterPass> {
  llvm::raw_ostream &os;

public:
  explicit CallGraphPrinterPass(llvm::raw_ostream &ostream)
      : impl::CallGraphPrinterPassBase<CallGraphPrinterPass>(), os(ostream) {}

protected:
  void runOnOperation() override {
    markAllAnalysesPreserved();

    auto &cga = getAnalysis<CallGraphAnalysis>();
    cga.getCallGraph().print(os);
  }
};

std::unique_ptr<mlir::Pass> createCallGraphPrinterPass(llvm::raw_ostream &os = llvm::errs()) {
  return std::make_unique<CallGraphPrinterPass>(os);
}

class CallGraphSCCsPrinterPass
    : public impl::CallGraphSCCsPrinterPassBase<CallGraphSCCsPrinterPass> {
  llvm::raw_ostream &os;

public:
  explicit CallGraphSCCsPrinterPass(llvm::raw_ostream &ostream)
      : impl::CallGraphSCCsPrinterPassBase<CallGraphSCCsPrinterPass>(), os(ostream) {}

protected:
  void runOnOperation() override {
    markAllAnalysesPreserved();

    auto &CG = getAnalysis<CallGraphAnalysis>();
    unsigned sccNum = 0;
    os << "SCCs for the program in PostOrder:";
    for (auto SCCI = llvm::scc_begin<const llzk::CallGraph *>(&CG.getCallGraph()); !SCCI.isAtEnd(); ++SCCI) {
      const std::vector<const CallGraphNode *> &nextSCC = *SCCI;
      os << "\nSCC #" << ++sccNum << ": ";
      bool First = true;
      for (const CallGraphNode *CGN : nextSCC) {
        if (First) {
          First = false;
        } else {
          os << ", ";
        }
        if (FuncOp fn = CGN->getCalledFunction()) {
          os << fn.getFullyQualifiedName();
        } else {
          os << "external node";
        }
      }

      if (nextSCC.size() == 1 && SCCI.hasCycle()) {
        os << " (Has self-loop).";
      }
    }
    os << '\n';
  }
};

std::unique_ptr<mlir::Pass> createCallGraphSCCsPrinterPass(llvm::raw_ostream &os = llvm::errs()) {
  return std::make_unique<CallGraphSCCsPrinterPass>(os);
}

} // namespace llzk