#pragma once

#include "llzk/Dialect/LLZK/Analysis/AbstractLatticeValue.h"
#include "llzk/Dialect/LLZK/Analysis/AnalysisWrappers.h"
#include "llzk/Dialect/LLZK/Analysis/ConstraintDependencyGraph.h"
#include "llzk/Dialect/LLZK/Analysis/DenseAnalysis.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/Compare.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/AnalysisManager.h>
#include <mlir/Support/LLVM.h>

#include <llvm/Support/SMTAPI.h>

namespace llzk {

/* Interval */

/// @brief The interval arms may be concrete values or symbolic values that
/// are dependent on inputs.
class Interval {
public:
  enum Type { TypeA, TypeB, TypeC, TypeF, Unbound };

  Interval() : ty(Unbound), a(), b() {}

  bool operator==(const Interval &rhs) const { return true; }
  Interval &operator+=(const Interval &rhs) { return *this; }

private:
  Type ty;
  llvm::APInt a, b;
};

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const Interval &i) {
  os << "todo";
  return os;
}

static_assert(dataflow::ScalarLatticeValue<Interval>, "foobar");

/* IntervalAnalysisLatticeValue */

class IntervalAnalysisLatticeValue
    : public dataflow::AbstractLatticeValue<IntervalAnalysisLatticeValue, Interval> {
public:
  IntervalAnalysisLatticeValue() : i(), expr(nullptr) {}

  explicit IntervalAnalysisLatticeValue(llvm::SMTExprRef exprRef) : i(), expr(exprRef) {}

private:
  Interval i;
  llvm::SMTExprRef expr;
};

/* IntervalAnalysisLattice */

class IntervalAnalysisLattice : public dataflow::AbstractDenseLattice {
public:
  using Value = IntervalAnalysisLatticeValue;
  using ValueMap = mlir::DenseMap<mlir::Value, Value>;

  using AbstractDenseLattice::AbstractDenseLattice;

  mlir::ChangeResult join(const AbstractDenseLattice &rhs) override {
    llvm::report_fatal_error("IntervalAnalysisLattice::join : todo!");
    return mlir::ChangeResult::NoChange;
  }

  mlir::ChangeResult meet(const AbstractDenseLattice &rhs) override {
    llvm::report_fatal_error("IntervalDataFlowAnalysis::meet : todo!");
    return mlir::ChangeResult::NoChange;
  }

  void print(mlir::raw_ostream &os) const override {}

  mlir::FailureOr<Value> getValue(mlir::Value v) const {
    auto it = valMap.find(v);
    if (it == valMap.end()) {
      return mlir::failure();
    }
    return it->second;
  }

private:
  ValueMap valMap;
};

/* IntervalDataFlowAnalysis */

class IntervalDataFlowAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<IntervalAnalysisLattice> {
  using Base = dataflow::DenseForwardDataFlowAnalysis<IntervalAnalysisLattice>;
  using Lattice = IntervalAnalysisLattice;
  using LatticeValue = IntervalAnalysisLattice::Value;

public:
  explicit IntervalDataFlowAnalysis(mlir::DataFlowSolver &solver)
      : Base::DenseForwardDataFlowAnalysis(solver), dataflowSolver(solver),
        smtSolver(llvm::CreateZ3Solver()) {}

  void visitCallControlFlowTransfer(
      mlir::CallOpInterface call, dataflow::CallControlFlowAction action, const Lattice &before,
      Lattice *after
  ) override {
    llvm::report_fatal_error("IntervalDataFlowAnalysis::visitCallControlFlowTransfer : todo!");
  }

  void visitOperation(mlir::Operation *op, const Lattice &before, Lattice *after) override {
    IntervalAnalysisLattice::ValueMap operandVals;

    auto constrainRefLattice = dataflowSolver.lookupState<ConstrainRefLattice>(op);
    ensure(constrainRefLattice, "failed to get lattice");

    for (auto &operand : op->getOpOperands()) {
      // We only care about felt type values.
      auto val = operand.get();
      auto latticeVal = before.getValue(val);
      if (mlir::succeeded(latticeVal)) {
        operandVals[val] = *latticeVal;
      } else {
        std::string symbolName;
        llvm::raw_string_ostream ss(symbolName);
        val.print(ss);

        auto expr = smtSolver->mkSymbol(symbolName.c_str(), smtSolver->getBitvectorSort(128));
        expr->print(llvm::errs());
        llvm::errs() << "\naha!\n";
        operandVals[val] = LatticeValue(expr);

        llvm::report_fatal_error("IntervalDataFlowAnalysis::visitOperation : todo!");
      }
    }
  }

private:
  void setToEntryState(Lattice *lattice) override {
    // the entry state is empty, so do nothing.
  }

  mlir::DataFlowSolver &dataflowSolver;
  llvm::SMTSolverRef smtSolver;
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

class StructIntervalAnalysis : public StructAnalysis<StructIntervals> {
public:
  using StructAnalysis::StructAnalysis;

  mlir::LogicalResult
  runAnalysis(mlir::DataFlowSolver &solver, mlir::AnalysisManager &moduleAnalysisManager) override {

    llvm::report_fatal_error("StructIntervalAnalysis::runAnalysis : todo!");
    return mlir::failure();
  }
};

/* ModuleIntervalAnalysis */

using ModuleIntervalAnalysis = ModuleAnalysis<
    StructIntervals, StructIntervalAnalysis, ConstrainRefAnalysis, IntervalDataFlowAnalysis>;

} // namespace llzk
