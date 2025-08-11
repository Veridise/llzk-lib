//===-- IntervalAnalysis.h --------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Analysis/AbstractLatticeValue.h"
#include "llzk/Analysis/AnalysisWrappers.h"
#include "llzk/Analysis/ConstraintDependencyGraph.h"
#include "llzk/Analysis/DenseAnalysis.h"
#include "llzk/Analysis/Field.h"
#include "llzk/Analysis/Intervals.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Util/APIntHelper.h"
#include "llzk/Util/Compare.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/AnalysisManager.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/MapVector.h>
#include <llvm/Support/SMTAPI.h>

#include <array>
#include <mutex>

namespace llzk {

/* ExpressionValue */

/// @brief Tracks a solver expression and an interval range for that expression.
/// Used as a scalar lattice value.
class ExpressionValue {
public:
  /* Must be default initializable to be a ScalarLatticeValue. */
  ExpressionValue() : i(), expr(nullptr) {}

  explicit ExpressionValue(const Field &f, llvm::SMTExprRef exprRef)
      : i(Interval::Entire(f)), expr(exprRef) {}

  ExpressionValue(const Field &f, llvm::SMTExprRef exprRef, llvm::APSInt singleVal)
      : i(Interval::Degenerate(f, singleVal)), expr(exprRef) {}

  ExpressionValue(llvm::SMTExprRef exprRef, Interval interval) : i(interval), expr(exprRef) {}

  llvm::SMTExprRef getExpr() const { return expr; }

  const Interval &getInterval() const { return i; }

  const Field &getField() const { return i.getField(); }

  /// @brief Return the current expression with a new interval.
  /// @param newInterval
  /// @return
  ExpressionValue withInterval(const Interval &newInterval) const {
    return ExpressionValue(expr, newInterval);
  }

  /* Required to be a ScalarLatticeValue. */
  /// @brief Fold two expressions together when overapproximating array elements.
  ExpressionValue &join(const ExpressionValue &rhs) {
    i = Interval::Entire(getField());
    return *this;
  }

  bool operator==(const ExpressionValue &rhs) const;

  /// @brief Compute the intersection of the lhs and rhs intervals, and create a solver
  /// expression that constrains both sides to be equal.
  /// @param solver
  /// @param lhs
  /// @param rhs
  /// @return
  friend ExpressionValue
  intersection(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  /// @brief Compute the union of the lhs and rhs intervals, and create a solver
  /// expression that constrains both sides to be equal.
  /// @param solver
  /// @param lhs
  /// @param rhs
  /// @return
  friend ExpressionValue
  join(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  // arithmetic ops

  friend ExpressionValue
  add(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  friend ExpressionValue
  sub(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  friend ExpressionValue
  mul(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  friend ExpressionValue
  div(llvm::SMTSolverRef solver, felt::DivFeltOp op, const ExpressionValue &lhs,
      const ExpressionValue &rhs);

  friend ExpressionValue
  mod(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  friend ExpressionValue
  cmp(llvm::SMTSolverRef solver, boolean::CmpOp op, const ExpressionValue &lhs,
      const ExpressionValue &rhs);

  friend ExpressionValue
  boolAnd(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  friend ExpressionValue
  boolOr(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  friend ExpressionValue
  boolXor(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  /// @brief Computes a solver expression based on the operation, but computes a fallback
  /// interval (which is just Entire, or unknown). Used for currently unsupported compute-only
  /// operations.
  /// @param solver
  /// @param op
  /// @param lhs
  /// @param rhs
  /// @return
  friend ExpressionValue fallbackBinaryOp(
      llvm::SMTSolverRef solver, mlir::Operation *op, const ExpressionValue &lhs,
      const ExpressionValue &rhs
  );

  friend ExpressionValue neg(llvm::SMTSolverRef solver, const ExpressionValue &val);

  friend ExpressionValue notOp(llvm::SMTSolverRef solver, const ExpressionValue &val);

  friend ExpressionValue boolNot(llvm::SMTSolverRef solver, const ExpressionValue &val);

  friend ExpressionValue
  fallbackUnaryOp(llvm::SMTSolverRef solver, mlir::Operation *op, const ExpressionValue &val);

  /* Utility */

  void print(mlir::raw_ostream &os) const;

  friend mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const ExpressionValue &e) {
    e.print(os);
    return os;
  }

  struct Hash {
    unsigned operator()(const ExpressionValue &e) const {
      return Interval::Hash {}(e.i) ^ llvm::hash_value(e.expr);
    }
  };

private:
  Interval i;
  llvm::SMTExprRef expr;
};

/* IntervalAnalysisLatticeValue */

class IntervalAnalysisLatticeValue
    : public dataflow::AbstractLatticeValue<IntervalAnalysisLatticeValue, ExpressionValue> {
public:
  using AbstractLatticeValue::AbstractLatticeValue;
};

/* IntervalAnalysisLattice */

class IntervalDataFlowAnalysis;

/// @brief Maps mlir::Values to LatticeValues.
///
class IntervalAnalysisLattice : public dataflow::AbstractDenseLattice {
public:
  using LatticeValue = IntervalAnalysisLatticeValue;
  // Map mlir::Values to LatticeValues
  using ValueMap = mlir::DenseMap<mlir::Value, LatticeValue>;
  // Map field references to LatticeValues. Used for field reads and writes.
  // Structure is component value -> field attribute -> latticeValue
  using FieldMap = mlir::DenseMap<mlir::Value, mlir::DenseMap<mlir::StringAttr, LatticeValue>>;
  // Expression to interval map for convenience.
  using ExpressionIntervals = mlir::DenseMap<llvm::SMTExprRef, Interval>;
  // Tracks all constraints and assignments in insertion order
  using ConstraintSet = llvm::SetVector<ExpressionValue>;

  using AbstractDenseLattice::AbstractDenseLattice;

  mlir::ChangeResult join(const AbstractDenseLattice &other) override;

  mlir::ChangeResult meet(const AbstractDenseLattice &rhs) override {
    llvm::report_fatal_error("IntervalDataFlowAnalysis::meet : unsupported");
    return mlir::ChangeResult::NoChange;
  }

  void print(mlir::raw_ostream &os) const override;

  mlir::FailureOr<LatticeValue> getValue(mlir::Value v) const;
  mlir::FailureOr<LatticeValue> getValue(mlir::Value v, mlir::StringAttr f) const;

  mlir::ChangeResult setValue(mlir::Value v, ExpressionValue e);
  mlir::ChangeResult setValue(mlir::Value v, mlir::StringAttr f, ExpressionValue e);

  mlir::ChangeResult addSolverConstraint(ExpressionValue e);

  friend mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const IntervalAnalysisLattice &l) {
    l.print(os);
    return os;
  }

  const ConstraintSet &getConstraints() const { return constraints; }

  mlir::FailureOr<Interval> findInterval(llvm::SMTExprRef expr) const;

  size_t size() const { return valMap.size(); }

  const ValueMap &getMap() const { return valMap; }

  ValueMap::iterator begin() { return valMap.begin(); }
  ValueMap::iterator end() { return valMap.end(); }
  ValueMap::const_iterator begin() const { return valMap.begin(); }
  ValueMap::const_iterator end() const { return valMap.end(); }

private:
  ValueMap valMap;
  FieldMap fieldMap;
  ConstraintSet constraints;
  ExpressionIntervals intervals;
};

/* IntervalDataFlowAnalysis */

class IntervalDataFlowAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<IntervalAnalysisLattice> {
  using Base = dataflow::DenseForwardDataFlowAnalysis<IntervalAnalysisLattice>;
  using Lattice = IntervalAnalysisLattice;
  using LatticeValue = IntervalAnalysisLattice::LatticeValue;

  // Map fields to their symbols
  using SymbolMap = mlir::DenseMap<ConstrainRef, llvm::SMTExprRef>;

public:
  explicit IntervalDataFlowAnalysis(
      mlir::DataFlowSolver &dataflowSolver, llvm::SMTSolverRef smt, const Field &f,
      bool propInputConstraints
  )
      : Base::DenseForwardDataFlowAnalysis(dataflowSolver), _dataflowSolver(dataflowSolver),
        smtSolver(smt), field(f), propagateInputConstraints(propInputConstraints) {}

  void visitCallControlFlowTransfer(
      mlir::CallOpInterface call, dataflow::CallControlFlowAction action, const Lattice &before,
      Lattice *after
  ) override;

  void visitOperation(mlir::Operation *op, const Lattice &before, Lattice *after) override;

  /// @brief Either return the existing SMT expression that corresponds to the ConstrainRef,
  /// or create one.
  /// @param r
  /// @return
  llvm::SMTExprRef getOrCreateSymbol(const ConstrainRef &r);

private:
  mlir::DataFlowSolver &_dataflowSolver;
  llvm::SMTSolverRef smtSolver;
  SymbolMap refSymbols;
  std::reference_wrapper<const Field> field;
  bool propagateInputConstraints;
  mlir::SymbolTableCollection tables;

  void setToEntryState(Lattice *lattice) override {
    // initial state should be empty, so do nothing here
  }

  llvm::SMTExprRef createFeltSymbol(const ConstrainRef &r) const;

  llvm::SMTExprRef createFeltSymbol(mlir::Value val) const;

  llvm::SMTExprRef createFeltSymbol(const char *name) const;

  bool isConstOp(mlir::Operation *op) const {
    return mlir::isa<
        felt::FeltConstantOp, mlir::arith::ConstantIndexOp, mlir::arith::ConstantIntOp>(op);
  }

  llvm::APSInt getConst(mlir::Operation *op) const;

  llvm::SMTExprRef createConstBitvectorExpr(llvm::APSInt v) const {
    return smtSolver->mkBitvector(v, field.get().bitWidth());
  }

  llvm::SMTExprRef createConstBoolExpr(bool v) const {
    return smtSolver->mkBitvector(mlir::APSInt((int)v), field.get().bitWidth());
  }

  bool isArithmeticOp(mlir::Operation *op) const {
    return mlir::isa<
        felt::AddFeltOp, felt::SubFeltOp, felt::MulFeltOp, felt::DivFeltOp, felt::ModFeltOp,
        felt::NegFeltOp, felt::InvFeltOp, felt::AndFeltOp, felt::OrFeltOp, felt::XorFeltOp,
        felt::NotFeltOp, felt::ShlFeltOp, felt::ShrFeltOp, boolean::CmpOp, boolean::AndBoolOp,
        boolean::OrBoolOp, boolean::XorBoolOp, boolean::NotBoolOp>(op);
  }

  ExpressionValue
  performBinaryArithmetic(mlir::Operation *op, const LatticeValue &a, const LatticeValue &b);

  ExpressionValue performUnaryArithmetic(mlir::Operation *op, const LatticeValue &a);

  /// @brief Recursively applies the new interval to the val's lattice value and to that value's
  /// operands, if possible. For example, if we know that X*Y is non-zero, then we know X and Y are
  /// non-zero, and can update X and Y's intervals accordingly.
  /// @param after The current lattice state. Assumes that this has already been joined with the
  /// `before` lattice in `visitOperation`, so lookups and updates can be performed on the `after`
  /// lattice alone.
  mlir::ChangeResult
  applyInterval(mlir::Operation *originalOp, Lattice *after, mlir::Value val, Interval newInterval);

  /// @brief Special handling for generalized (s - c0) * (s - c1) * ... * (s - cN) = 0 patterns.
  mlir::FailureOr<std::pair<llvm::DenseSet<mlir::Value>, Interval>> getGeneralizedDecompInterval(
      const ConstrainRefLattice *constrainRefLattice, mlir::Value lhs, mlir::Value rhs
  );

  bool isBoolOp(mlir::Operation *op) const {
    return mlir::isa<boolean::AndBoolOp, boolean::OrBoolOp, boolean::XorBoolOp, boolean::NotBoolOp>(
        op
    );
  }

  bool isConversionOp(mlir::Operation *op) const {
    return mlir::isa<cast::IntToFeltOp, cast::FeltToIndexOp>(op);
  }

  bool isApplyMapOp(mlir::Operation *op) const { return mlir::isa<polymorphic::ApplyMapOp>(op); }

  bool isAssertOp(mlir::Operation *op) const { return mlir::isa<boolean::AssertOp>(op); }

  bool isReadOp(mlir::Operation *op) const {
    return mlir::isa<component::FieldReadOp, polymorphic::ConstReadOp, array::ReadArrayOp>(op);
  }

  bool isWriteOp(mlir::Operation *op) const {
    return mlir::isa<component::FieldWriteOp, array::WriteArrayOp, array::InsertArrayOp>(op);
  }

  bool isArrayLengthOp(mlir::Operation *op) const { return mlir::isa<array::ArrayLengthOp>(op); }

  bool isEmitOp(mlir::Operation *op) const {
    return mlir::isa<constrain::EmitEqualityOp, constrain::EmitContainmentOp>(op);
  }

  bool isCreateOp(mlir::Operation *op) const {
    return mlir::isa<component::CreateStructOp, array::CreateArrayOp>(op);
  }

  bool isExtractArrayOp(mlir::Operation *op) const { return mlir::isa<array::ExtractArrayOp>(op); }

  bool isDefinitionOp(mlir::Operation *op) const {
    return mlir::isa<
        component::StructDefOp, function::FuncDefOp, component::FieldDefOp, global::GlobalDefOp,
        mlir::ModuleOp>(op);
  }

  bool isCallOp(mlir::Operation *op) const { return mlir::isa<function::CallOp>(op); }

  bool isReturnOp(mlir::Operation *op) const { return mlir::isa<function::ReturnOp>(op); }

  /// @brief Used for sanity checking and warnings about the analysis. If new operations
  /// are introduced and encountered, we can use this (and related methods) to issue
  /// warnings to users.
  /// @param op
  /// @return
  bool isConsideredOp(mlir::Operation *op) const {
    return isConstOp(op) || isArithmeticOp(op) || isBoolOp(op) || isConversionOp(op) ||
           isApplyMapOp(op) || isAssertOp(op) || isReadOp(op) || isWriteOp(op) ||
           isArrayLengthOp(op) || isEmitOp(op) || isCreateOp(op) || isDefinitionOp(op) ||
           isCallOp(op) || isReturnOp(op) || isExtractArrayOp(op);
  }
};

/* StructIntervals */

/// @brief Parameters and shared objects to pass to child analyses.
struct IntervalAnalysisContext {
  IntervalDataFlowAnalysis *intervalDFA;
  llvm::SMTSolverRef smtSolver;
  std::reference_wrapper<const Field> field;
  bool propagateInputConstraints;

  llvm::SMTExprRef getSymbol(const ConstrainRef &r) { return intervalDFA->getOrCreateSymbol(r); }
  const Field &getField() const { return field.get(); }
  bool doInputConstraintPropagation() const { return propagateInputConstraints; }
};

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
      mlir::ModuleOp mod, component::StructDefOp s, mlir::DataFlowSolver &solver,
      IntervalAnalysisContext &ctx
  ) {
    StructIntervals si(mod, s);
    if (si.computeIntervals(solver, ctx).failed()) {
      return mlir::failure();
    }
    return si;
  }

  mlir::LogicalResult computeIntervals(mlir::DataFlowSolver &solver, IntervalAnalysisContext &ctx);

  void print(mlir::raw_ostream &os, bool withConstraints = false, bool printCompute = false) const;

  const llvm::MapVector<ConstrainRef, Interval> &getConstrainIntervals() const {
    return constrainFieldRanges;
  }

  const llvm::SetVector<ExpressionValue> getConstrainSolverConstraints() const {
    return constrainSolverConstraints;
  }

  const llvm::MapVector<ConstrainRef, Interval> &getComputeIntervals() const {
    return computeFieldRanges;
  }

  const llvm::SetVector<ExpressionValue> getComputeSolverConstraints() const {
    return computeSolverConstraints;
  }

  friend mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const StructIntervals &si) {
    si.print(os);
    return os;
  }

private:
  mlir::ModuleOp mod;
  component::StructDefOp structDef;
  llvm::SMTSolverRef smtSolver;
  // llvm::MapVector keeps insertion order for consistent iteration
  llvm::MapVector<ConstrainRef, Interval> constrainFieldRanges, computeFieldRanges;
  // llvm::SetVector for the same reasons as above
  llvm::SetVector<ExpressionValue> constrainSolverConstraints, computeSolverConstraints;

  StructIntervals(mlir::ModuleOp m, component::StructDefOp s) : mod(m), structDef(s) {}
};

/* StructIntervalAnalysis */

class ModuleIntervalAnalysis;

class StructIntervalAnalysis : public StructAnalysis<StructIntervals, IntervalAnalysisContext> {
public:
  using StructAnalysis::StructAnalysis;
  virtual ~StructIntervalAnalysis() = default;

  mlir::LogicalResult runAnalysis(
      mlir::DataFlowSolver &solver, mlir::AnalysisManager &_, IntervalAnalysisContext &ctx
  ) override {
    auto computeRes = StructIntervals::compute(getModule(), getStruct(), solver, ctx);
    if (mlir::failed(computeRes)) {
      return mlir::failure();
    }
    setResult(std::move(*computeRes));
    return mlir::success();
  }
};

/* ModuleIntervalAnalysis */

class ModuleIntervalAnalysis
    : public ModuleAnalysis<StructIntervals, IntervalAnalysisContext, StructIntervalAnalysis> {

public:
  ModuleIntervalAnalysis(mlir::Operation *op)
      : ModuleAnalysis(op), smtSolver(llvm::CreateZ3Solver()), field(std::nullopt) {}
  virtual ~ModuleIntervalAnalysis() = default;

  void setField(const Field &f) { field = f; }
  void setPropagateInputConstraints(bool prop) { propagateInputConstraints = prop; }

protected:
  void initializeSolver() override {
    ensure(field.has_value(), "field not set, could not generate analysis context");
    (void)solver.load<ConstrainRefAnalysis>();
    auto smtSolverRef = smtSolver;
    bool prop = propagateInputConstraints;
    intervalDFA = solver.load<IntervalDataFlowAnalysis, llvm::SMTSolverRef, const Field &, bool>(
        std::move(smtSolverRef), field.value(), std::move(prop)
    );
  }

  IntervalAnalysisContext getContext() override {
    ensure(field.has_value(), "field not set, could not generate analysis context");
    return {
        .intervalDFA = intervalDFA,
        .smtSolver = smtSolver,
        .field = field.value(),
        .propagateInputConstraints = propagateInputConstraints,
    };
  }

private:
  llvm::SMTSolverRef smtSolver;
  IntervalDataFlowAnalysis *intervalDFA;
  std::optional<std::reference_wrapper<const Field>> field;
  bool propagateInputConstraints;
};

} // namespace llzk

namespace llvm {

template <> struct DenseMapInfo<llzk::ExpressionValue> {

  static SMTExprRef getEmptyExpr() {
    static auto emptyPtr = reinterpret_cast<SMTExprRef>(1);
    return emptyPtr;
  }
  static SMTExprRef getTombstoneExpr() {
    static auto tombstonePtr = reinterpret_cast<SMTExprRef>(2);
    return tombstonePtr;
  }

  static llzk::ExpressionValue getEmptyKey() {
    return llzk::ExpressionValue(llzk::Field::getField("bn128"), getEmptyExpr());
  }
  static inline llzk::ExpressionValue getTombstoneKey() {
    return llzk::ExpressionValue(llzk::Field::getField("bn128"), getTombstoneExpr());
  }
  static unsigned getHashValue(const llzk::ExpressionValue &e) {
    return llzk::ExpressionValue::Hash {}(e);
  }
  static bool isEqual(const llzk::ExpressionValue &lhs, const llzk::ExpressionValue &rhs) {
    if (lhs.getExpr() == getEmptyExpr() || lhs.getExpr() == getTombstoneExpr() ||
        rhs.getExpr() == getEmptyExpr() || rhs.getExpr() == getTombstoneExpr()) {
      return lhs.getExpr() == rhs.getExpr();
    }
    return lhs == rhs;
  }
};

} // namespace llvm
