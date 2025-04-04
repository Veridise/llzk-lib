//===- DenseAnalysis.cpp - Dense data-flow analysis -------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from mlir/lib/Analysis/DataFlow/DenseAnalysis.cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/Analysis/DenseAnalysis.h"
#include "llzk/Dialect/LLZK/Util/ErrorHelper.h"
#include "llzk/Dialect/LLZK/Util/SymbolHelper.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>

#include <cassert>
#include <optional>

using namespace mlir;
using namespace mlir::dataflow;

namespace llzk {
namespace dataflow {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

void markAllOpsAsLive(mlir::DataFlowSolver &solver, mlir::Operation *top) {
  for (mlir::Region &region : top->getRegions()) {
    for (mlir::Block &block : region) {
      (void)solver.getOrCreateState<mlir::dataflow::Executable>(&block)->setToLive();
      for (mlir::Operation &oper : block) {
        markAllOpsAsLive(solver, &oper);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// AbstractDenseForwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

LogicalResult AbstractDenseForwardDataFlowAnalysis::initialize(Operation *top) {
  // Visit every operation and block.
  processOperation(top);
  for (Region &region : top->getRegions()) {
    for (Block &block : region) {
      visitBlock(&block);
      for (Operation &op : block) {
        if (failed(initialize(&op))) {
          return failure();
        }
      }
    }
  }
  return success();
}

LogicalResult AbstractDenseForwardDataFlowAnalysis::visit(ProgramPoint point) {
  if (auto *op = llvm::dyn_cast_if_present<Operation *>(point)) {
    processOperation(op);
  } else if (auto *block = llvm::dyn_cast_if_present<Block *>(point)) {
    visitBlock(block);
  } else {
    return failure();
  }
  return success();
}

/// LLZK: This function has been modified to use LLZK symbol helpers instead of
/// the built-in resolveCallable method.
void AbstractDenseForwardDataFlowAnalysis::visitCallOperation(
    CallOpInterface call, const AbstractDenseLattice &before, AbstractDenseLattice *after
) {
  // Allow for customizing the behavior of calls to external symbols, including
  // when the analysis is explicitly marked as non-interprocedural.
  auto callable = resolveCallable<FuncOp>(tables, call);
  if (!getSolverConfig().isInterprocedural() ||
      (mlir::succeeded(callable) && !callable->get().getCallableRegion())) {
    return visitCallControlFlowTransfer(call, CallControlFlowAction::ExternalCallee, before, after);
  }

  /// LLZK: The PredecessorState Analysis state does not work for LLZK's custom calls.
  /// We therefore accumulate predecessor operations (return ops) manually.
  SmallVector<Operation *> predecessors;
  auto fnOp = callable.value().get();
  fnOp.walk([&predecessors](ReturnOp ret) mutable { predecessors.push_back(ret); });

  // If we have no predecessors, we cannot reason about dataflow, since there is
  // no return value.
  if (predecessors.empty()) {
    return setToEntryState(after);
  }

  for (Operation *predecessor : predecessors) {
    // Get the lattices at callee return:
    //
    //   func.func @callee() {
    //     ...
    //     return  // predecessor
    //     // latticeAtCalleeReturn
    //   }
    //   func.func @caller() {
    //     ...
    //     call @callee
    //     // latticeAfterCall
    //     ...
    //   }
    AbstractDenseLattice *latticeAfterCall = after;
    const AbstractDenseLattice *latticeAtCalleeReturn =
        getLatticeFor(call.getOperation(), predecessor);
    visitCallControlFlowTransfer(
        call, CallControlFlowAction::ExitCallee, *latticeAtCalleeReturn, latticeAfterCall
    );
  }
}

void AbstractDenseForwardDataFlowAnalysis::processOperation(Operation *op) {
  // If the containing block is not executable, bail out.
  if (!getOrCreateFor<Executable>(op, op->getBlock())->isLive()) {
    return;
  }

  // Get the dense lattice to update.
  AbstractDenseLattice *after = getLattice(op);

  // Get the dense state before the execution of the op.
  const AbstractDenseLattice *before;
  if (Operation *prev = op->getPrevNode()) {
    before = getLatticeFor(op, prev);
  } else {
    before = getLatticeFor(op, op->getBlock());
  }

  // If this op implements region control-flow, then control-flow dictates its
  // transfer function.
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    return visitRegionBranchOperation(op, branch, after);
  }

  // If this is a call operation, then join its lattices across known return
  // sites.
  if (auto call = dyn_cast<CallOpInterface>(op)) {
    return visitCallOperation(call, *before, after);
  }

  // Invoke the operation transfer function.
  visitOperationImpl(op, *before, after);
}

/// LLZK: Removing use of PredecessorState because it does not work with LLZK's
/// CallOp and FuncOp definitions.
void AbstractDenseForwardDataFlowAnalysis::visitBlock(Block *block) {
  // If the block is not executable, bail out.
  if (!getOrCreateFor<Executable>(block, block)->isLive()) {
    return;
  }

  // Get the dense lattice to update.
  AbstractDenseLattice *after = getLattice(block);

  // The dense lattices of entry blocks are set by region control-flow or the
  // callgraph.
  if (block->isEntryBlock()) {
    // Check if this block is the entry block of a callable region.
    auto callable = dyn_cast<CallableOpInterface>(block->getParentOp());
    if (callable && callable.getCallableRegion() == block->getParent()) {
      if (!getSolverConfig().isInterprocedural()) {
        return setToEntryState(after);
      }
      /// LLZK: Get callsites of the callable as the predecessors.
      auto moduleOpRes = getTopRootModule(callable.getOperation());
      ensure(mlir::succeeded(moduleOpRes), "could not get root module from callable");

      SmallVector<Operation *> callsites;
      moduleOpRes->walk([this, &callable, &callsites](CallOp call) mutable {
        auto calledFnRes = resolveCallable<FuncOp>(tables, call);
        if (mlir::succeeded(calledFnRes) &&
            calledFnRes->get().getCallableRegion() == callable.getCallableRegion()) {
          callsites.push_back(call);
        }
      });

      for (Operation *callsite : callsites) {
        // Get the dense lattice before the callsite.
        const AbstractDenseLattice *before;
        if (Operation *prev = callsite->getPrevNode()) {
          before = getLatticeFor(block, prev);
        } else {
          before = getLatticeFor(block, callsite->getBlock());
        }

        visitCallControlFlowTransfer(
            cast<CallOpInterface>(callsite), CallControlFlowAction::EnterCallee, *before, after
        );
      }
      return;
    }

    // Check if we can reason about the control-flow.
    if (auto branch = dyn_cast<RegionBranchOpInterface>(block->getParentOp())) {
      return visitRegionBranchOperation(block, branch, after);
    }

    // Otherwise, we can't reason about the data-flow.
    return setToEntryState(after);
  }

  // Join the state with the state after the block's predecessors.
  for (Block::pred_iterator it = block->pred_begin(), e = block->pred_end(); it != e; ++it) {
    // Skip control edges that aren't executable.
    Block *predecessor = *it;
    if (!getOrCreateFor<Executable>(block, getProgramPoint<CFGEdge>(predecessor, block))
             ->isLive()) {
      continue;
    }

    // Merge in the state from the predecessor's terminator.
    join(after, *getLatticeFor(block, predecessor->getTerminator()));
  }
}

void AbstractDenseForwardDataFlowAnalysis::visitRegionBranchOperation(
    ProgramPoint point, RegionBranchOpInterface branch, AbstractDenseLattice *after
) {
  // Get the terminator predecessors.
  const auto *predecessors = getOrCreateFor<PredecessorState>(point, point);
  assert(predecessors->allPredecessorsKnown() && "unexpected unresolved region successors");

  for (Operation *op : predecessors->getKnownPredecessors()) {
    const AbstractDenseLattice *before;
    // If the predecessor is the parent, get the state before the parent.
    if (op == branch) {
      if (Operation *prev = op->getPrevNode()) {
        before = getLatticeFor(point, prev);
      } else {
        before = getLatticeFor(point, op->getBlock());
      }

      // Otherwise, get the state after the terminator.
    } else {
      before = getLatticeFor(point, op);
    }

    // This function is called in two cases:
    //   1. when visiting the block (point = block);
    //   2. when visiting the parent operation (point = parent op).
    // In both cases, we are looking for predecessor operations of the point,
    //   1. predecessor may be the terminator of another block from another
    //   region (assuming that the block does belong to another region via an
    //   assertion) or the parent (when parent can transfer control to this
    //   region);
    //   2. predecessor may be the terminator of a block that exits the
    //   region (when region transfers control to the parent) or the operation
    //   before the parent.
    // In the latter case, just perform the join as it isn't the control flow
    // affected by the region.
    std::optional<unsigned> regionFrom =
        op == branch ? std::optional<unsigned>() : op->getBlock()->getParent()->getRegionNumber();
    if (auto *toBlock = point.dyn_cast<Block *>()) {
      unsigned regionTo = toBlock->getParent()->getRegionNumber();
      visitRegionBranchControlFlowTransfer(branch, regionFrom, regionTo, *before, after);
    } else {
      assert(point.get<Operation *>() == branch && "expected to be visiting the branch itself");
      // Only need to call the arc transfer when the predecessor is the region
      // or the op itself, not the previous op.
      if (op->getParentOp() == branch || op == branch) {
        visitRegionBranchControlFlowTransfer(
            branch, regionFrom, /*regionTo=*/std::nullopt, *before, after
        );
      } else {
        join(after, *before);
      }
    }
  }
}

const AbstractDenseLattice *
AbstractDenseForwardDataFlowAnalysis::getLatticeFor(ProgramPoint dependent, ProgramPoint point) {
  AbstractDenseLattice *state = getLattice(point);
  addDependency(state, dependent);
  return state;
}

} // namespace dataflow
} // namespace llzk
