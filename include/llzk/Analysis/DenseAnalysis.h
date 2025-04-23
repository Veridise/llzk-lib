//===-- DenseAnalysis.h - Dense data-flow analysis --------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from mlir/include/mlir/Analysis/DataFlow/DenseAnalysis.h.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements (LLZK-tailored) dense data-flow analysis using the
/// data-flow analysis framework. The analysis is forward and conditional and
/// uses the results of dead code analysis to prune dead code during the
/// analysis.
///
/// This file has been ported from the MLIR dense analysis so that it may be
/// tailored to work for LLZK modules,
/// as LLZK modules have different symbol lookup mechanisms that are currently
/// incompatible with the builtin MLIR dataflow analyses.
/// This file is mostly left as original in MLIR, with notes added where
/// changes have been made.
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

namespace llzk {
namespace dataflow {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// LLZK: Added this utility to ensure analysis is performed for all structs
/// in a given module.
///
/// @brief Mark all operations from the top and included in the top operation
/// as live so the solver will perform dataflow analyses.
/// @param solver The solver.
/// @param top The top-level operation.
void markAllOpsAsLive(mlir::DataFlowSolver &solver, mlir::Operation *top);

//===----------------------------------------------------------------------===//
// AbstractDenseForwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

using AbstractDenseLattice = mlir::dataflow::AbstractDenseLattice;
using CallControlFlowAction = mlir::dataflow::CallControlFlowAction;

/// LLZK: This class has been ported from the MLIR DenseAnalysis utilities to
/// allow for the use of custom LLZK symbol lookup logic. The class has been
/// left as unmodified as possible, with explicit comments added where modifications
/// have been made.
///
/// Base class for dense forward data-flow analyses. Dense data-flow analysis
/// attaches a lattice between the execution of operations and implements a
/// transfer function from the lattice before each operation to the lattice
/// after. The lattice contains information about the state of the program at
/// that point.
///
/// In this implementation, a lattice attached to an operation represents the
/// state of the program after its execution, and a lattice attached to block
/// represents the state of the program right before it starts executing its
/// body.
class AbstractDenseForwardDataFlowAnalysis : public mlir::DataFlowAnalysis {
public:
  using mlir::DataFlowAnalysis::DataFlowAnalysis;

  /// Initialize the analysis by visiting every program point whose execution
  /// may modify the program state; that is, every operation and block.
  mlir::LogicalResult initialize(mlir::Operation *top) override;

  /// Visit a program point that modifies the state of the program. If this is a
  /// block, then the state is propagated from control-flow predecessors or
  /// callsites. If this is a call operation or region control-flow operation,
  /// then the state after the execution of the operation is set by control-flow
  /// or the callgraph. Otherwise, this function invokes the operation transfer
  /// function.
  mlir::LogicalResult visit(mlir::ProgramPoint point) override;

protected:
  /// Propagate the dense lattice before the execution of an operation to the
  /// lattice after its execution.
  virtual void visitOperationImpl(
      mlir::Operation *op, const AbstractDenseLattice &before, AbstractDenseLattice *after
  ) = 0;

  /// Get the dense lattice after the execution of the given program point.
  virtual AbstractDenseLattice *getLattice(mlir::ProgramPoint point) = 0;

  /// Get the dense lattice after the execution of the given program point and
  /// add it as a dependency to a program point. That is, every time the lattice
  /// after point is updated, the dependent program point must be visited, and
  /// the newly triggered visit might update the lattice after dependent.
  const AbstractDenseLattice *getLatticeFor(mlir::ProgramPoint dependent, mlir::ProgramPoint point);

  /// Set the dense lattice at control flow entry point and propagate an update
  /// if it changed.
  virtual void setToEntryState(AbstractDenseLattice *lattice) = 0;

  /// Join a lattice with another and propagate an update if it changed.
  void join(AbstractDenseLattice *lhs, const AbstractDenseLattice &rhs) {
    propagateIfChanged(lhs, lhs->join(rhs));
  }

  /// Visit an operation. If this is a call operation or region control-flow
  /// operation, then the state after the execution of the operation is set by
  /// control-flow or the callgraph. Otherwise, this function invokes the
  /// operation transfer function.
  virtual void processOperation(mlir::Operation *op);

  /// Propagate the dense lattice forward along the control flow edge from
  /// `regionFrom` to `regionTo` regions of the `branch` operation. `nullopt`
  /// values correspond to control flow branches originating at or targeting the
  /// `branch` operation itself. Default implementation just joins the states,
  /// meaning that operations implementing `RegionBranchOpInterface` don't have
  /// any effect on the lattice that isn't already expressed by the interface
  /// itself.
  virtual void visitRegionBranchControlFlowTransfer(
      mlir::RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
      std::optional<unsigned> regionTo, const AbstractDenseLattice &before,
      AbstractDenseLattice *after
  ) {
    join(after, before);
  }

  /// Propagate the dense lattice forward along the call control flow edge,
  /// which can be either entering or exiting the callee. Default implementation
  /// for enter and exit callee actions just meets the states, meaning that
  /// operations implementing `CallOpInterface` don't have any effect on the
  /// lattice that isn't already expressed by the interface itself. Default
  /// implementation for the external callee action additionally sets the
  /// "after" lattice to the entry state.
  virtual void visitCallControlFlowTransfer(
      mlir::CallOpInterface call, CallControlFlowAction action, const AbstractDenseLattice &before,
      AbstractDenseLattice *after
  ) {
    join(after, before);
    // Note that `setToEntryState` may be a "partial fixpoint" for some
    // lattices, e.g., lattices that are lists of maps of other lattices will
    // only set fixpoint for "known" lattices.
    if (action == CallControlFlowAction::ExternalCallee) {
      setToEntryState(after);
    }
  }

  /// Visit a program point within a region branch operation with predecessors
  /// in it. This can either be an entry block of one of the regions of the
  /// parent operation itself.
  void visitRegionBranchOperation(
      mlir::ProgramPoint point, mlir::RegionBranchOpInterface branch, AbstractDenseLattice *after
  );

  /// LLZK: Added for use of symbol helper caching.
  mlir::SymbolTableCollection tables;

private:
  /// Visit a block. The state at the start of the block is propagated from
  /// control-flow predecessors or callsites.
  void visitBlock(mlir::Block *block);

  /// Visit an operation for which the data flow is described by the
  /// `CallOpInterface`.
  void visitCallOperation(
      mlir::CallOpInterface call, const AbstractDenseLattice &before, AbstractDenseLattice *after
  );
};

//===----------------------------------------------------------------------===//
// DenseForwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// LLZK: This class has been ported so that it can inherit from our port of
/// the AbstractDenseForwardDataFlowAnalysis above. It is otherwise left unchanged.
///
/// A dense forward data-flow analysis for propagating lattices before and
/// after the execution of every operation across the IR by implementing
/// transfer functions for operations.
///
/// `LatticeT` is expected to be a subclass of `AbstractDenseLattice`.
template <typename LatticeT>
class DenseForwardDataFlowAnalysis : public AbstractDenseForwardDataFlowAnalysis {
  static_assert(
      std::is_base_of<AbstractDenseLattice, LatticeT>::value,
      "analysis state class expected to subclass AbstractDenseLattice"
  );

public:
  using AbstractDenseForwardDataFlowAnalysis::AbstractDenseForwardDataFlowAnalysis;

  /// Visit an operation with the dense lattice before its execution. This
  /// function is expected to set the dense lattice after its execution and
  /// trigger change propagation in case of change.
  virtual void visitOperation(mlir::Operation *op, const LatticeT &before, LatticeT *after) = 0;

  /// Hook for customizing the behavior of lattice propagation along the call
  /// control flow edges. Two types of (forward) propagation are possible here:
  ///   - `action == CallControlFlowAction::Enter` indicates that:
  ///     - `before` is the state before the call operation;
  ///     - `after` is the state at the beginning of the callee entry block;
  ///   - `action == CallControlFlowAction::Exit` indicates that:
  ///     - `before` is the state at the end of a callee exit block;
  ///     - `after` is the state after the call operation.
  /// By default, the `after` state is simply joined with the `before` state.
  /// Concrete analyses can override this behavior or delegate to the parent
  /// call for the default behavior. Specifically, if the `call` op may affect
  /// the lattice prior to entering the callee, the custom behavior can be added
  /// for `action == CallControlFlowAction::Enter`. If the `call` op may affect
  /// the lattice post exiting the callee, the custom behavior can be added for
  /// `action == CallControlFlowAction::Exit`.
  virtual void visitCallControlFlowTransfer(
      mlir::CallOpInterface call, CallControlFlowAction action, const LatticeT &before,
      LatticeT *after
  ) {
    AbstractDenseForwardDataFlowAnalysis::visitCallControlFlowTransfer(call, action, before, after);
  }

  /// Hook for customizing the behavior of lattice propagation along the control
  /// flow edges between regions and their parent op. The control flows from
  /// `regionFrom` to `regionTo`, both of which may be `nullopt` to indicate the
  /// parent op. The lattice is propagated forward along this edge. The lattices
  /// are as follows:
  ///   - `before:`
  ///     - if `regionFrom` is a region, this is the lattice at the end of the
  ///       block that exits the region; note that for multi-exit regions, the
  ///       lattices are equal at the end of all exiting blocks, but they are
  ///       associated with different program points.
  ///     - otherwise, this is the lattice before the parent op.
  ///   - `after`:
  ///     - if `regionTo` is a region, this is the lattice at the beginning of
  ///       the entry block of that region;
  ///     - otherwise, this is the lattice after the parent op.
  /// By default, the `after` state is simply joined with the `before` state.
  /// Concrete analyses can override this behavior or delegate to the parent
  /// call for the default behavior. Specifically, if the `branch` op may affect
  /// the lattice before entering any region, the custom behavior can be added
  /// for `regionFrom == nullopt`. If the `branch` op may affect the lattice
  /// after all terminated, the custom behavior can be added for `regionTo ==
  /// nullptr`. The behavior can be further refined for specific pairs of "from"
  /// and "to" regions.
  virtual void visitRegionBranchControlFlowTransfer(
      mlir::RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
      std::optional<unsigned> regionTo, const LatticeT &before, LatticeT *after
  ) {
    AbstractDenseForwardDataFlowAnalysis::visitRegionBranchControlFlowTransfer(
        branch, regionFrom, regionTo, before, after
    );
  }

protected:
  /// Get the dense lattice after this program point.
  LatticeT *getLattice(mlir::ProgramPoint point) override { return getOrCreate<LatticeT>(point); }

  /// Set the dense lattice at control flow entry point and propagate an update
  /// if it changed.
  virtual void setToEntryState(LatticeT *lattice) = 0;
  void setToEntryState(AbstractDenseLattice *lattice) override {
    setToEntryState(static_cast<LatticeT *>(lattice));
  }

  /// Type-erased wrappers that convert the abstract dense lattice to a derived
  /// lattice and invoke the virtual hooks operating on the derived lattice.
  void visitOperationImpl(
      mlir::Operation *op, const AbstractDenseLattice &before, AbstractDenseLattice *after
  ) final {
    visitOperation(op, static_cast<const LatticeT &>(before), static_cast<LatticeT *>(after));
  }
  void visitCallControlFlowTransfer(
      mlir::CallOpInterface call, CallControlFlowAction action, const AbstractDenseLattice &before,
      AbstractDenseLattice *after
  ) final {
    visitCallControlFlowTransfer(
        call, action, static_cast<const LatticeT &>(before), static_cast<LatticeT *>(after)
    );
  }
  void visitRegionBranchControlFlowTransfer(
      mlir::RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
      std::optional<unsigned> regionTo, const AbstractDenseLattice &before,
      AbstractDenseLattice *after
  ) final {
    visitRegionBranchControlFlowTransfer(
        branch, regionFrom, regionTo, static_cast<const LatticeT &>(before),
        static_cast<LatticeT *>(after)
    );
  }
};

} // namespace dataflow
} // namespace llzk
