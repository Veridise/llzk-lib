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

#include "llzk/Analysis/DenseAnalysis.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Util/ErrorHelper.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>

#include <optional>

using namespace mlir;
using Executable = mlir::dataflow::Executable;
using CFGEdge = mlir::dataflow::CFGEdge;

namespace llzk {

using namespace function;

namespace dataflow {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

void markAllOpsAsLive(DataFlowSolver &solver, Operation *top) {
  for (Region &region : top->getRegions()) {
    for (Block &block : region) {
      ProgramPoint *point = solver.getProgramPointBefore(&block);
      (void)solver.getOrCreateState<Executable>(point)->setToLive();
      for (Operation &oper : block) {
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
  if (failed(processOperation(top))) {
    return failure();
  }
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

LogicalResult AbstractDenseForwardDataFlowAnalysis::visit(ProgramPoint *point) {
  if (!point->isBlockStart()) {
    return processOperation(point->getPrevOp());
  }
  visitBlock(point->getBlock());
  return success();
}

/// LLZK: This function has been modified to use LLZK symbol helpers instead of
/// the built-in resolveCallable method.
void AbstractDenseForwardDataFlowAnalysis::visitCallOperation(
    CallOpInterface call, const AbstractDenseLattice &before, AbstractDenseLattice *after
) {
  // Allow for customizing the behavior of calls to external symbols, including
  // when the analysis is explicitly marked as non-interprocedural.
  auto callable = resolveCallable<FuncDefOp>(tables, call);
  if (!getSolverConfig().isInterprocedural() ||
      (succeeded(callable) && !callable->get().getCallableRegion())) {
    return visitCallControlFlowTransfer(call, CallControlFlowAction::ExternalCallee, before, after);
  }

  /// LLZK: The PredecessorState Analysis state does not work for LLZK's custom calls.
  /// We therefore accumulate predecessor operations (return ops) manually.
  SmallVector<Operation *> predecessors;
  callable->get().walk([&predecessors](ReturnOp ret) mutable { predecessors.push_back(ret); });

  // If we have no predecessors, we cannot reason about dataflow, since there is
  // no return value.
  if (predecessors.empty()) {
    return setToEntryState(after);
  }

  for (Operation *predecessor : predecessors) {
    // Get the lattices at callee return:
    //
    //   function.def @callee() {
    //     ...
    //     return  // predecessor
    //     // latticeAtCalleeReturn
    //   }
    //   function.def @caller() {
    //     ...
    //     call @callee
    //     // latticeAfterCall
    //     ...
    //   }
    AbstractDenseLattice *latticeAfterCall = after;
    const AbstractDenseLattice *latticeAtCalleeReturn =
        getLatticeFor(getProgramPointAfter(call.getOperation()), getProgramPointAfter(predecessor));
    visitCallControlFlowTransfer(
        call, CallControlFlowAction::ExitCallee, *latticeAtCalleeReturn, latticeAfterCall
    );
  }
}

LogicalResult AbstractDenseForwardDataFlowAnalysis::processOperation(Operation *op) {
  ProgramPoint *point = getProgramPointAfter(op);
  // If the containing block is not executable, bail out.
  if (op->getBlock() != nullptr &&
      !getOrCreateFor<Executable>(point, getProgramPointBefore(op->getBlock()))->isLive()) {
    return success();
  }

  // Get the dense lattice to update.
  AbstractDenseLattice *after = getLattice(point);

  // Get the dense state before the execution of the op.
  const AbstractDenseLattice *before = getLatticeFor(point, getProgramPointBefore(op));

  // If this op implements region control-flow, then control-flow dictates its
  // transfer function.
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    visitRegionBranchOperation(point, branch, after);
    return success();
  }

  // If this is a call operation, then join its lattices across known return
  // sites.
  if (auto call = dyn_cast<CallOpInterface>(op)) {
    visitCallOperation(call, *before, after);
    return success();
  }

  // Invoke the operation transfer function.
  return visitOperationImpl(op, *before, after);
}

/// LLZK: Removing use of PredecessorState because it does not work with LLZK's
/// CallOp and FuncDefOp definitions.
void AbstractDenseForwardDataFlowAnalysis::visitBlock(Block *block) {
  // If the block is not executable, bail out.
  ProgramPoint *point = getProgramPointBefore(block);
  if (!getOrCreateFor<Executable>(point, point)->isLive()) {
    return;
  }

  // Get the dense lattice to update.
  AbstractDenseLattice *after = getLattice(point);

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
      ensure(succeeded(moduleOpRes), "could not get root module from callable");
      SmallVector<Operation *> callsites;
      moduleOpRes->walk([this, &callable, &callsites](CallOp call) mutable {
        auto calledFnRes = resolveCallable<FuncDefOp>(tables, call);
        if (succeeded(calledFnRes) &&
            calledFnRes->get().getCallableRegion() == callable.getCallableRegion()) {
          callsites.push_back(call);
        }
      });

      for (Operation *callsite : callsites) {
        // Get the dense lattice before the callsite.
        const AbstractDenseLattice *before = getLatticeFor(point, getProgramPointBefore(callsite));

        visitCallControlFlowTransfer(
            llvm::cast<CallOpInterface>(callsite), CallControlFlowAction::EnterCallee, *before,
            after
        );
      }
      return;
    }

    // Check if we can reason about the control-flow.
    if (auto branch = dyn_cast<RegionBranchOpInterface>(block->getParentOp())) {
      return visitRegionBranchOperation(point, branch, after);
    }

    // Otherwise, we can't reason about the data-flow.
    return setToEntryState(after);
  }

  // Join the state with the state after the block's predecessors.
  for (Block::pred_iterator it = block->pred_begin(), e = block->pred_end(); it != e; ++it) {
    // Skip control edges that aren't executable.
    Block *predecessor = *it;
    if (!getOrCreateFor<Executable>(point, getLatticeAnchor<CFGEdge>(predecessor, block))
             ->isLive()) {
      continue;
    }

    // Merge in the state from the predecessor's terminator.
    join(after, *getLatticeFor(point, getProgramPointAfter(predecessor->getTerminator())));
  }
}

/// LLZK: Removing use of PredecessorState because it does not work with LLZK's lookup logic.
void AbstractDenseForwardDataFlowAnalysis::visitRegionBranchOperation(
    ProgramPoint *point, RegionBranchOpInterface branch, AbstractDenseLattice *after
) {
  Operation *op = point->isBlockStart() ? point->getBlock()->getParentOp() : point->getPrevOp();
  if (op) {
    const AbstractDenseLattice *before;
    // If the predecessor is the parent, get the state before the parent.
    if (op == branch) {
      before = getLatticeFor(point, getProgramPointBefore(op));
      // Otherwise, get the state after the terminator.
    } else {
      before = getLatticeFor(point, getProgramPointAfter(op));
    }

    // This function is called in two cases:
    //   1. when visiting the block (point = block start);
    //   2. when visiting the parent operation (point = iter after parent op).
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
    if (point->isBlockStart()) {
      unsigned regionTo = point->getBlock()->getParent()->getRegionNumber();
      visitRegionBranchControlFlowTransfer(branch, regionFrom, regionTo, *before, after);
    } else {
      assert(point->getPrevOp() == branch && "expected to be visiting the branch itself");
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
AbstractDenseForwardDataFlowAnalysis::getLatticeFor(ProgramPoint *dependent, LatticeAnchor anchor) {
  AbstractDenseLattice *state = getLattice(anchor);
  addDependency(state, dependent);
  return state;
}

} // namespace dataflow

} // namespace llzk
