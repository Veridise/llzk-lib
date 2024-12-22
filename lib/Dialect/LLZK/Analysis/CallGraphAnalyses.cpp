/**
 * The contents of this file are adapted from llvm/lib/Analysis/CallGraph.cpp
 */
#include "llzk/Dialect/LLZK/Analysis/CallGraphAnalyses.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"

#include <llvm/ADT/DepthFirstIterator.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>

namespace llzk {

CallGraphAnalysis::CallGraphAnalysis(mlir::Operation *op) : cg(nullptr) {
  if (auto modOp = mlir::dyn_cast<mlir::ModuleOp>(op)) {
    cg = std::make_unique<llzk::CallGraph>(modOp);
  } else {
    auto error_message = "CallGraphAnalysis expects provided op to be a ModuleOp!";
    op->emitError(error_message);
    llvm::report_fatal_error(error_message);
  }
}

CallGraphReachabilityAnalysis::CallGraphReachabilityAnalysis(
    mlir::Operation *op, mlir::AnalysisManager &am
)
    : callGraph(am.getAnalysis<CallGraphAnalysis>().getCallGraph()) {
  if (!mlir::isa<mlir::ModuleOp>(op)) {
    auto error_message = "CallGraphReachabilityAnalysis expects provided op to be a ModuleOp!";
    op->emitError(error_message);
    llvm::report_fatal_error(error_message);
  }
}

bool CallGraphReachabilityAnalysis::isReachable(FuncOp &A, FuncOp &B) const {
  if (isReachableCached(A, B)) {
    return true;
  }

  auto startNode = callGraph.get().lookupNode(A.getCallableRegion());
  /**
   * NOTE: This is a potential cause of performance issues, as some circuits
   * may perform poorly for DFS. However, we don't have enough examples at this
   * time to demonstrate such an issue, so we will stick with DFS for simplicity
   * for now. If performance issues arise, it may be beneficial to switch to
   * inverse DFS or a different algorithm entirely.
   */
  auto dfsIt = llvm::df_begin<const CallGraphNode *>(startNode);
  auto dfsEnd = llvm::df_end<const CallGraphNode *>(startNode);
  for (; dfsIt != dfsEnd; ++dfsIt) {
    const CallGraphNode *currNode = *dfsIt;
    if (currNode->isExternal()) {
      continue;
    }
    FuncOp currFn = currNode->getCalledFunction();

    // Update the cache according to the path before checking if B is reachable.
    for (unsigned i = 0; i < dfsIt.getPathLength(); i++) {
      FuncOp ancestorFn = dfsIt.getPath(i)->getCalledFunction();
      reachabilityMap[ancestorFn].insert(currFn);
    }

    if (isReachableCached(currFn, B)) {
      return true;
    }
  }
  return false;
}

} // namespace llzk