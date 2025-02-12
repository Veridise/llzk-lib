#pragma once

#include "llzk/Dialect/LLZK/Analysis/AbstractLatticeValue.h"
#include "llzk/Dialect/LLZK/Analysis/DenseAnalysis.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/Compare.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/AnalysisManager.h>
#include <mlir/Support/LLVM.h>

namespace llzk {

/* Interval */

class Interval {
public:
  bool operator==(const Interval &rhs) const { return true; }
  Interval &operator+=(const Interval &rhs) { return *this; }
};

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const Interval &i) {
  os << "todo";
  return os;
}

static_assert(dataflow::ScalarLatticeValue<Interval>, "foobar");

/* IntervalAnalysisLatticeValue */

class IntervalAnalysisLatticeValue
    : public dataflow::AbstractLatticeValue<IntervalAnalysisLatticeValue, Interval> {};

/* IntervalAnalysisLattice */

class IntervalAnalysisLattice : public dataflow::AbstractDenseLattice {
public:
  using AbstractDenseLattice::AbstractDenseLattice;

  mlir::ChangeResult join(const AbstractDenseLattice &rhs) override {
    return mlir::ChangeResult::NoChange;
  }

  mlir::ChangeResult meet(const AbstractDenseLattice &rhs) override {
    return mlir::ChangeResult::NoChange;
  }

  void print(mlir::raw_ostream &os) const override {}
};

/* IntervalDataFlowAnalysis */

class IntervalDataFlowAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<IntervalAnalysisLattice> {
  using Base = dataflow::DenseForwardDataFlowAnalysis<IntervalAnalysisLattice>;
  using Lattice = IntervalAnalysisLattice;

public:
  using Base::DenseForwardDataFlowAnalysis;

  void visitCallControlFlowTransfer(
      mlir::CallOpInterface call, dataflow::CallControlFlowAction action, const Lattice &before,
      Lattice *after
  ) override {}

  void visitOperation(mlir::Operation *op, const Lattice &before, Lattice *after) override {}

private:
  void setToEntryState(Lattice *lattice) override {
    // the entry state is empty, so do nothing.
  }
};

/* StructIntervals */

class StructIntervals {
public:
  /// @brief Compute the struct intervals.
  /// @param mod The LLZK-complaint module that is the parent of struct `s`.
  /// @param s The struct to compute value intervals for.
  /// @param solver A pre-configured DataFlowSolver. The liveness of the struct must
  /// already be computed in this solver in order for the analysis to run.
  /// @param am A module-level analysis manager. This analysis manager needs to originate
  /// from a module-level analysis (i.e., for the `mod` module) so that analyses
  /// for other constraints can be queried via the getChildAnalysis method.
  /// @return
  static mlir::FailureOr<StructIntervals> compute(
      mlir::ModuleOp mod, StructDefOp s, mlir::DataFlowSolver &solver, mlir::AnalysisManager &am
  ) {
    StructIntervals si(mod, s);
    if (si.computeIntervals(solver, am).failed()) {
      return mlir::failure();
    }
    return si;
  }

  StructIntervals(mlir::ModuleOp m, StructDefOp s) : mod(m), structDef(s) {}

  mlir::LogicalResult computeIntervals(mlir::DataFlowSolver &solver, mlir::AnalysisManager &am) {
    return mlir::failure();
  }

private:
  mlir::ModuleOp mod;
  StructDefOp structDef;
};

/* StructIntervalAnalysis */

class StructIntervalAnalysis {
public:
  StructIntervalAnalysis(mlir::Operation *op, mlir::AnalysisManager &am) {}

  mlir::LogicalResult
  constructIntervals(mlir::DataFlowSolver &solver, mlir::AnalysisManager &moduleAnalysisManager) {}

  std::shared_ptr<StructIntervals> getIntervalsPtr() const { return intervals; }

private:
  std::shared_ptr<StructIntervals> intervals;
};

/* ModuleIntervalAnalysis */

class ModuleIntervalAnalysis {
  /// Using a map, not an unordered map, to control sorting order for iteration.
  using DependencyMap =
      std::map<StructDefOp, std::shared_ptr<StructIntervals>, OpLocationLess<StructDefOp>>;

public:
  ModuleIntervalAnalysis(mlir::Operation *op, mlir::AnalysisManager &am) {
    if (auto modOp = mlir::dyn_cast<mlir::ModuleOp>(op)) {
      mlir::DataFlowConfig config;
      mlir::DataFlowSolver solver(config);
      dataflow::markAllOpsAsLive(solver, modOp);

      // The analysis is run at the module level so that lattices are computed
      // for global functions as well.
      solver.load<IntervalDataFlowAnalysis>();
      auto res = solver.initializeAndRun(modOp);
      debug::ensure(res.succeeded(), "solver failed to run on module!");

      llvm::report_fatal_error("todo!");

      modOp.walk([this, &solver, &am](StructDefOp s) {
        auto &ca = am.getChildAnalysis<StructIntervalAnalysis>(s);
        if (mlir::failed(ca.constructIntervals(solver, am))) {
          auto error_message =
              "StructIntervalAnalysis failed to compute intervals for " + mlir::Twine(s.getName());
          s->emitError(error_message);
          llvm::report_fatal_error(error_message);
        }
        dependencies[s] = ca.getIntervalsPtr();
      });
    } else {
      auto error_message = "ModuleIntervalAnalysis expects provided op to be an mlir::ModuleOp!";
      op->emitError(error_message);
      llvm::report_fatal_error(error_message);
    }
  }

private:
  DependencyMap dependencies;
};

} // namespace llzk
                   