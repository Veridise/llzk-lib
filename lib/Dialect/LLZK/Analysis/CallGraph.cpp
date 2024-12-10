/**
 * The contents of this file are adapted from llvm/lib/Analysis/CallGraph.cpp
 */
#include "llzk/Dialect/LLZK/Analysis/CallGraph.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>

namespace llzk {

using namespace ::mlir;

/* CallGraph */

CallGraph::CallGraph(ModuleOp M) : M(M), EntryNode(getOrInsertFunction(FuncOp(nullptr))) {
  // Add every interesting function to the call graph.
  M.walk([&](FuncOp F) {
    addToCallGraph(F);
  });
}

CallGraph::CallGraph(CallGraph &&Arg)
    : M(Arg.M), FunctionMap(std::move(Arg.FunctionMap)),
      EntryNode(Arg.EntryNode) {
  Arg.FunctionMap.clear();

  // Update parent CG for all call graph's nodes.
  EntryNode->CG = this;
  for (auto &P : FunctionMap) {
    P.second->CG = this;
  }
}

CallGraph::~CallGraph() {
// Reset all node's use counts to zero before deleting them to prevent an
// assertion from firing.
#ifndef NDEBUG
  for (auto &I : FunctionMap) {
    I.second->allReferencesDropped();
  }
#endif
}

void CallGraph::addToCallGraph(FuncOp &F) {
  CallGraphNode *Node = getOrInsertFunction(F);
  // TODO: Main component logic.
  if (F.getName() == FUNC_NAME_COMPUTE || F.getName() == FUNC_NAME_CONSTRAIN) {
    EntryNode->addCalledFunction(nullptr, Node);
  }
  populateCallGraphNode(Node);
}

void CallGraph::populateCallGraphNode(CallGraphNode *Node) {
  FuncOp F = Node->getFunction();

  // Look for calls by this function.
  F->walk([&](CallOp callOp) {
    auto calledFnSym = callOp.getCallee();
    FuncOp calledFn = FuncOp(mlir::SymbolTable::lookupSymbolIn(M, calledFnSym));
    assert(calledFn != nullptr && "Should be able to find all function!");
    Node->addCalledFunction(&callOp, getOrInsertFunction(calledFn));
  });
}

void CallGraph::print(mlir::raw_ostream &OS) const {
  // Print in a deterministic order by sorting CallGraphNodes by name.  We do
  // this here to avoid slowing down the non-printing fast path.

  llvm::SmallVector<CallGraphNode *, 16> Nodes;
  Nodes.reserve(FunctionMap.size());

  for (const auto &I : *this)
    Nodes.push_back(I.second.get());

  llvm::sort(Nodes, [](CallGraphNode *LHS, CallGraphNode *RHS) {
    if (LHS->getFunction() && RHS->getFunction()) {
      return LHS->getFunction() < RHS->getFunction();
    }
    return RHS->getFunction() != nullptr;
  });

  for (CallGraphNode *CN : Nodes) {
    CN->print(OS);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void CallGraph::dump() const { print(llvm::dbgs()); }
#endif

// removeFunctionFromModule - Unlink the function from this module, returning
// it.  Because this removes the function from the module, the call graph node
// is destroyed.  This is only valid if the function does not call any other
// functions (ie, there are no edges in it's CGN).  The easiest way to do this
// is to dropAllReferences before calling this.
//
FuncOp CallGraph::removeFunctionFromModule(CallGraphNode *CGN) {
  assert(CGN->empty() && "Cannot remove function from call "
         "graph if it references other functions!");
  FuncOp F = CGN->getFunction(); // Get the function for the call graph node
  FunctionMap.erase(F);          // Remove the call graph node from the map

  F->erase();
  return F;
}

// getOrInsertFunction - This method is identical to calling operator[], but
// it will insert a new CallGraphNode for the specified function if one does
// not already exist.
CallGraphNode *CallGraph::getOrInsertFunction(FuncOp F) {
  auto &CGN = FunctionMap[F];
  if (CGN) {
    return CGN.get();
  }

  auto containedInModule = [&] (mlir::Operation *op, FuncOp &f) {
    assert(f);
    if (op->hasTrait<mlir::OpTrait::SymbolTable>()) {
      return mlir::SymbolTable::lookupSymbolIn(op, f.getName()) != nullptr;
    }
    return op == f;
  };
  bool isContained = false;
  if (F) {
    M.walk([&](mlir::Operation *op) {
      isContained = containedInModule(op, F);
      if (isContained) {
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
  }
  // null checks work on mlir custom dialect ops
  assert((!F || isContained) && "Function not in current module!");
  CGN = std::make_unique<CallGraphNode>(this, F);
  return CGN.get();
}

bool CallGraph::isReachable(FuncOp &A, FuncOp &B) const {
  if (FunctionMap.find(A) == FunctionMap.end() || FunctionMap.find(B) == FunctionMap.end()) {
    return false;
  }
  const CallGraphNode *start = FunctionMap.at(A).get(), *end = FunctionMap.at(B).get();

  if (cachedReachability.find(end) != cachedReachability.end()) {
    const auto &s = cachedReachability.at(end);
    if (s.find(start) != s.end()) {
      return true;
    }
    // We don't return false here because our cached results might come from
    // a separate ancestry path.
  }

  /*
    Simple BFS until we:
    1. Reach the end pointer, or
    2. Explore all nodes from start and find nothing.
  */
  std::deque<const CallGraphNode *> frontier;
  std::unordered_set<const CallGraphNode *> visited;
  frontier.push_back(start);
  while (!frontier.empty()) {
    const CallGraphNode *n = frontier.front();
    frontier.pop_front();

    if (visited.find(n) != visited.end()) {
      continue;
    }
    visited.insert(n);

    for (const auto &[_, child] : *n) {
      const auto &parentSet = cachedReachability[n]; // default construct in empty case
      auto &childSet = cachedReachability[child];
      // update cache, then check, then add to frontier
      childSet.insert(n); // add parent
      childSet.insert(parentSet.begin(), parentSet.end()); // add parent's set

      if (child == end) {
        return true;
      }

      frontier.push_back(child);
    }
  }

  return false;
}

/* CallGraphNode */

void CallGraphNode::print(mlir::raw_ostream &OS) const {
  if (F) {
    OS << "Call graph node for function: '" << F->getName() << "'";
  } else {
    OS << "Entry call graph node (null function)";
  }

  OS << "<<" << this << ">>  #uses=" << getNumReferences() << '\n';

  for (const auto &I : *this) {
    OS << "  CS<" << I.first;
    if (I.first) {
      OS << " (" << *I.first << ")";
    }
    OS << "> calls ";
    if (const FuncOp &FI = I.second->getFunction()) {
      OS << "function '" << FI->getName() <<"'\n";
    } else {
      OS << "external node\n";
    }
  }
  OS << '\n';
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void CallGraphNode::dump() const { print(llvm::dbgs()); }
#endif

/// removeCallEdgeFor - This method removes the edge in the node for the
/// specified call site.  Note that this method takes linear time, so it
/// should be used sparingly.
void CallGraphNode::removeCallEdgeFor(CallOp *Call) {
  for (CalledFunctionsVector::iterator I = CalledFunctions.begin(); ; ++I) {
    assert(I != CalledFunctions.end() && "Cannot find callsite to remove!");
    if (I->first == Call) {
      I->second->DropRef();
      *I = CalledFunctions.back();
      CalledFunctions.pop_back();

      // Remove all references to callback functions if there are any.
      FuncOp op = FuncOp(mlir::SymbolTable::lookupSymbolIn(CG->getModule(), Call->getCallee()));
      removeOneAbstractEdgeTo(CG->getOrInsertFunction(op));
      return;
    }
  }
}

// removeAnyCallEdgeTo - This method removes any call edges from this node to
// the specified callee function.  This takes more time to execute than
// removeCallEdgeTo, so it should not be used unless necessary.
void CallGraphNode::removeAnyCallEdgeTo(CallGraphNode *Callee) {
  for (unsigned i = 0, e = CalledFunctions.size(); i != e; ++i)
    if (CalledFunctions[i].second == Callee) {
      Callee->DropRef();
      CalledFunctions[i] = CalledFunctions.back();
      CalledFunctions.pop_back();
      --i; --e;
    }
}

/// removeOneAbstractEdgeTo - Remove one edge associated with a null callsite
/// from this node to the specified callee function.
void CallGraphNode::removeOneAbstractEdgeTo(CallGraphNode *Callee) {
  for (CalledFunctionsVector::iterator I = CalledFunctions.begin(); ; ++I) {
    assert(I != CalledFunctions.end() && "Cannot find callee to remove!");
    CallRecord &CR = *I;
    if (CR.second == Callee && !CR.first) {
      Callee->DropRef();
      *I = CalledFunctions.back();
      CalledFunctions.pop_back();
      return;
    }
  }
}

/// replaceCallEdge - This method replaces the edge in the node for the
/// specified call site with a new one.  Note that this method takes linear
/// time, so it should be used sparingly.
void CallGraphNode::replaceCallEdge(CallOp *Call, CallOp *NewCall,
                                    CallGraphNode *NewNode) {
  for (CalledFunctionsVector::iterator I = CalledFunctions.begin(); ; ++I) {
    assert(I != CalledFunctions.end() && "Cannot find callsite to remove!");
    if (I->first == Call) {
      I->second->DropRef();
      I->first = NewCall;
      I->second = NewNode;
      NewNode->AddRef();

      // Refresh callback references. Do not resize CalledFunctions if the
      // number of callbacks is the same for new and old call sites.
      SmallVector<CallGraphNode *, 4u> OldCBs;
      SmallVector<CallGraphNode *, 4u> NewCBs;
      FuncOp oldCB = FuncOp(mlir::SymbolTable::lookupSymbolIn(CG->getModule(), Call->getCallee()));
      auto oldNode = CG->getOrInsertFunction(oldCB);

      for (auto J = CalledFunctions.begin();; ++J) {
        assert(J != CalledFunctions.end() && "Cannot find callsite to update!");
        if (!J->first && J->second == oldNode) {
          J->second = NewNode;
          oldNode->DropRef();
          NewNode->AddRef();
          break;
        }
      }

      return;
    }
  }
}

/* Passes and Analyses */

CallGraphAnalysis::CallGraphAnalysis(mlir::Operation *op) : cg(nullptr) {
  if (auto modOp = mlir::dyn_cast<mlir::ModuleOp>(op)) {
    cg = std::make_unique<CallGraph>(modOp);
  } else {
    auto error_message = "CallGraphAnalysis expects provided op to be a ModuleOp!";
    op->emitError(error_message);
    llvm::report_fatal_error(error_message);
  }
}

void CallGraphPrinterPass::runOnOperation() {
  markAllAnalysesPreserved();

  auto &cga = getAnalysis<CallGraphAnalysis>();
  cga.getCallGraph().print(OS);
}

void CallGraphSCCsPrinterPass::runOnOperation() {
  markAllAnalysesPreserved();

  auto &CG = getAnalysis<CallGraphAnalysis>();
  unsigned sccNum = 0;
  OS << "SCCs for the program in PostOrder:";
  for (auto SCCI = llvm::scc_begin(&CG.getCallGraph()); !SCCI.isAtEnd(); ++SCCI) {
    const std::vector<CallGraphNode *> &nextSCC = *SCCI;
    OS << "\nSCC #" << ++sccNum << ": ";
    bool First = true;
    for (CallGraphNode *CGN : nextSCC) {
      if (First) {
        First = false;
      } else {
        OS << ", ";
      }
      OS << (CGN->getFunction() ? CGN->getFunction()->getName().getStringRef()
                                : "external node");
    }

    if (nextSCC.size() == 1 && SCCI.hasCycle()) {
      OS << " (Has self-loop).";
    }
  }
  OS << "\n";
}

} // namespace llzk