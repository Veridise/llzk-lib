#pragma once

#include "llzk/Dialect/LLZK/Analysis/AbstractLatticeValue.h"
#include "llzk/Dialect/LLZK/Analysis/AnalysisWrappers.h"
#include "llzk/Dialect/LLZK/Analysis/ConstraintDependencyGraph.h"
#include "llzk/Dialect/LLZK/Analysis/DenseAnalysis.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/Compare.h"
#include "llzk/Dialect/LLZK/Util/OpHelper.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/AnalysisManager.h>
#include <mlir/Support/LLVM.h>

#include <llvm/Support/SMTAPI.h>

#include <array>

namespace llzk {

/* Field */

/// TODO: Need a global configuration for this.
class Field {
public:
  static const Field &BN128() {
    static constexpr size_t bn128_nbits = 254;
    static constexpr auto bn128_str =
        "21888242871839275222246405745257275088696311157297823662689037894645226208583";
    static auto bn128_prime = llvm::APInt(bn128_nbits, bn128_str, 10);
    static auto bn128_half =
        (bn128_prime + llvm::APInt(bn128_nbits, 1, 10)).udiv(llvm::APInt(bn128_nbits, 2, 10));

    static Field singleton(bn128_prime, bn128_half);
    return singleton;
  }

  llvm::APInt prime() const { return primeMod; }

  llvm::APInt half() const { return halfPrime; }

  llvm::APInt reduce(llvm::APInt i) const { return i.urem(prime()); }

  size_t bitWidth() const { return primeMod.getBitWidth(); }

  friend bool operator==(const Field &lhs, const Field &rhs) {
    return lhs.primeMod == rhs.primeMod;
  }

private:
  Field(llvm::APInt p, llvm::APInt h) : primeMod(p), halfPrime(h) {}

  llvm::APInt primeMod, halfPrime;
};

/* UnreducedInterval */

class Interval;

class UnreducedInterval {
public:
  UnreducedInterval(llvm::APInt x, llvm::APInt y) : a(x), b(y) {}

  Interval reduce(const Field &field) const;

  llvm::APInt width() const { return b - a; }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const UnreducedInterval &ui) {
    os << "Unreduced:[ " << ui.a << ", " << ui.b << " ]";
    return os;
  }

  UnreducedInterval intersect(const UnreducedInterval &rhs) const;

  UnreducedInterval doUnion(const UnreducedInterval &rhs) const;

  /* Arithmetic operations */

  friend UnreducedInterval operator+(const UnreducedInterval &lhs, const UnreducedInterval &rhs);

private:
  llvm::APInt a, b;
};

/* Interval */

/// @brief The interval arms may be concrete values or symbolic values that
/// are dependent on inputs.
class Interval {
public:
  enum class Type { TypeA = 0, TypeB, TypeC, TypeF, Empty, Degenerate, Entire };
  static constexpr std::array<std::string_view, 7> TypeNames = {"TypeA", "TypeB", "TypeC",
                                                                "TypeF", "Empty", "Degenerate",
                                                                "Entire"};

  static std::string_view TypeName(Type t) { return TypeNames.at(static_cast<size_t>(t)); }

  static Interval Empty() {
    static Interval empty(Type::Empty);
    return empty;
  }

  bool isEmpty() const { return ty == Type::Empty; }

  static Interval Entire() {
    static Interval empty(Type::Entire);
    return empty;
  }

  bool isEntire() const { return ty == Type::Entire; }

  static Interval Degenerate(llvm::APInt val) { return Interval(Type::Degenerate, val, val); }

  bool isDegenerate() const { return ty == Type::Degenerate; }

  static Interval TypeA(llvm::APInt a, llvm::APInt b) { return Interval(Type::TypeA, a, b); }

  static Interval TypeB(llvm::APInt a, llvm::APInt b) { return Interval(Type::TypeB, a, b); }

  static Interval TypeC(llvm::APInt a, llvm::APInt b) { return Interval(Type::TypeC, a, b); }

  static Interval TypeF(llvm::APInt a, llvm::APInt b) { return Interval(Type::TypeF, a, b); }

  UnreducedInterval toUnreduced() const { return UnreducedInterval(a, b); }

  UnreducedInterval firstUnreduced() const {
    if (isOneOf<Type::TypeF>()) {
      return UnreducedInterval(field.get().prime() - a, b);
    }
    return toUnreduced();
  }

  UnreducedInterval secondUnreduced() const {
    ensure(isOneOf<Type::TypeA, Type::TypeB, Type::TypeC>(), "unsupported range type");
    return UnreducedInterval(a - field.get().prime(), b - field.get().prime());
  }

  bool operator==(const Interval &rhs) const { return ty == rhs.ty && a == rhs.a && b == rhs.b; }

  // To satisfy the dataflow::ScalarLatticeValue requirements, this class must
  // be default initializable. The default interval is the full range of values.
  Interval() : ty(Type::Entire), a(), b(), field(Field::BN128()) {}

  template <Type... Types> bool isOneOf() const { return ((ty == Types) || ...); }

  template <std::pair<Type, Type>... Pairs>
  static bool areOneOf(const Interval &a, const Interval &b) {
    return ((a.ty == std::get<0>(Pairs) && b.ty == std::get<1>(Pairs)) || ...);
  }

  Interval &join(const Interval &rhs) {
    llvm::report_fatal_error("todo");
    return *this;
  }

  void print(mlir::raw_ostream &os) const {
    os << TypeName(ty);
    if (isOneOf<Type::Degenerate>()) {
      os << '(' << a << ')';
    } else if (!isOneOf<Type::Entire, Type::Empty>()) {
      os << ":[ " << a << ", " << b << " ]";
    }
  }

  /// Union
  Interval join(const Interval &rhs) const;

  /// Intersect
  Interval intersect(const Interval &rhs) const;

  struct Hash {
    unsigned operator()(const Interval &i) const {
      return std::hash<const Field *> {}(&i.field.get()) ^ std::hash<Type> {}(i.ty) ^
             llvm::hash_value(i.a) ^ llvm::hash_value(i.b);
    }
  };

  /* arithmetic ops */

  friend Interval operator+(const Interval &lhs, const Interval &rhs);

  /* debug */

  friend mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const Interval &i) {
    i.print(os);
    return os;
  }

private:
  explicit Interval(Type t) : field(Field::BN128()), ty(t), a(), b() {}
  Interval(Type t, llvm::APInt lhs, llvm::APInt rhs)
      : field(Field::BN128()), ty(t), a(lhs.zext(field.get().bitWidth())),
        b(rhs.zext(field.get().bitWidth())) {}

  std::reference_wrapper<const Field> field;
  Type ty;
  llvm::APInt a, b;
};

static_assert(dataflow::ScalarLatticeValue<Interval>, "foobar");

/* ExpressionValue */

class ExpressionValue {
public:
  ExpressionValue() : i(), expr(nullptr) {}

  explicit ExpressionValue(llvm::SMTExprRef exprRef) : i(Interval::Entire()), expr(exprRef) {}

  ExpressionValue(llvm::SMTExprRef exprRef, llvm::APInt singleVal)
      : i(Interval::Degenerate(singleVal)), expr(exprRef) {}

  ExpressionValue(llvm::SMTExprRef exprRef, Interval interval) : i(interval), expr(exprRef) {}

  llvm::SMTExprRef getExpr() const { return expr; }

  const Interval &getInterval() const { return i; }

  bool operator==(const ExpressionValue &rhs) const {
    if (expr == nullptr && rhs.expr == nullptr) {
      return i == rhs.i;
    }
    if (expr == nullptr || rhs.expr == nullptr) {
      return false;
    }
    return i == rhs.i && *expr == *rhs.expr;
  }

  ExpressionValue &join(const ExpressionValue &rhs) {
    llvm::report_fatal_error("not yet supported!");
    return *this;
  }

  friend ExpressionValue
  intersection(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
    Interval res = lhs.i.intersect(rhs.i);
    auto exprEq = solver->mkEqual(lhs.expr, rhs.expr);
    return ExpressionValue(exprEq, res);
  }

  void print(mlir::raw_ostream &os) const {
    if (expr) {
      expr->print(os);
    } else {
      os << "<null expression>";
    }

    os << " ( interval: " << i << " )";
  }

  // arithmetic ops

  friend ExpressionValue
  add(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
    ExpressionValue res;
    res.i = lhs.i + rhs.i;
    res.expr = solver->mkBVAdd(lhs.expr, rhs.expr);
    return res;
  }

  // debug

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
  // Expression to interval map for convenience.
  using ExpressionIntervals = mlir::DenseMap<llvm::SMTExprRef, Interval>;
  // Tracks all constraints and assignments
  using ConstraintSet = mlir::DenseSet<ExpressionValue>;

  using AbstractDenseLattice::AbstractDenseLattice;

  mlir::ChangeResult join(const AbstractDenseLattice &other) override {
    const auto *rhs = dynamic_cast<const IntervalAnalysisLattice *>(&other);
    if (!rhs) {
      llvm::report_fatal_error("invalid join lattice type");
    }
    mlir::ChangeResult res = mlir::ChangeResult::NoChange;
    for (auto &[k, v] : rhs->valMap) {
      auto it = valMap.find(k);
      if (it == valMap.end() || it->second != v) {
        valMap[k] = v;
        res |= mlir::ChangeResult::Change;
      }
    }
    for (auto &v : rhs->constraints) {
      if (constraints.find(v) == constraints.end()) {
        constraints.insert(v);
        res |= mlir::ChangeResult::Change;
      }
    }
    for (auto &[e, i] : rhs->intervals) {
      auto it = intervals.find(e);
      if (it == intervals.end() || it->second != i) {
        intervals[e] = i;
        res |= mlir::ChangeResult::Change;
      }
    }
    return res;
  }

  mlir::ChangeResult meet(const AbstractDenseLattice &rhs) override {
    llvm::report_fatal_error("IntervalDataFlowAnalysis::meet : todo!");
    return mlir::ChangeResult::NoChange;
  }

  void print(mlir::raw_ostream &os) const override {
    os << "IntervalAnalysisLattice { ";
    for (auto &[ref, val] : valMap) {
      os << "\n    (valMap) " << ref << " := " << val;
    }
    for (auto &[expr, interval] : intervals) {
      os << "\n    (intervals) ";
      expr->print(os);
      os << " in " << interval;
    }
    if (!valMap.empty()) {
      os << '\n';
    }
    os << '}';
  }

  mlir::FailureOr<LatticeValue> getValue(mlir::Value v) const {
    auto it = valMap.find(v);
    if (it == valMap.end()) {
      return mlir::failure();
    }
    return it->second;
  }

  mlir::ChangeResult setValue(mlir::Value v, ExpressionValue e) {
    LatticeValue val(e);
    if (valMap[v] == val) {
      return mlir::ChangeResult::NoChange;
    }
    valMap[v] = val;
    intervals[e.getExpr()] = e.getInterval();

    llvm::errs() << __FUNCTION__ << ": set ";
    e.getExpr()->print(llvm::errs());
    llvm::errs() << " to " << e.getInterval() << "\n";

    return mlir::ChangeResult::Change;
  }

  mlir::ChangeResult addSolverConstraint(ExpressionValue e) {
    if (constraints.find(e) == constraints.end()) {
      constraints.insert(e);
      return mlir::ChangeResult::Change;
    }
    return mlir::ChangeResult::NoChange;
  }

  friend mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const IntervalAnalysisLattice &l) {
    l.print(os);
    return os;
  }

  const ConstraintSet &getConstraints() const { return constraints; }

  mlir::FailureOr<Interval> findInterval(llvm::SMTExprRef expr) const {
    auto it = intervals.find(expr);
    if (it != intervals.end()) {
      return it->second;
    }
    return mlir::failure();
  }

private:
  ValueMap valMap;
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
  explicit IntervalDataFlowAnalysis(mlir::DataFlowSolver &solver, llvm::SMTSolverRef smt)
      : Base::DenseForwardDataFlowAnalysis(solver), dataflowSolver(solver), smtSolver(smt) {}

  void visitCallControlFlowTransfer(
      mlir::CallOpInterface call, dataflow::CallControlFlowAction action, const Lattice &before,
      Lattice *after
  ) override {
    llvm::report_fatal_error("IntervalDataFlowAnalysis::visitCallControlFlowTransfer : todo!");
  }

  void visitOperation(mlir::Operation *op, const Lattice &before, Lattice *after) override {

    llvm::SmallVector<LatticeValue> operandVals;

    llvm::errs() << "before lattice: " << before << "\n";

    auto constrainRefLattice = dataflowSolver.lookupState<ConstrainRefLattice>(op);
    ensure(constrainRefLattice, "failed to get lattice");

    for (auto &operand : op->getOpOperands()) {
      auto val = operand.get();

      // First, lookup the operand value in the before state.
      auto priorState = before.getValue(val);
      if (mlir::succeeded(priorState)) {
        llvm::errs() << "  before: " << val << " := " << *priorState << "\n";
        operandVals.push_back(*priorState);
        continue;
      }
      llvm::errs() << "  before: no val for " << val << "\n";
      // Else, look up the stored value by constrain ref.
      // We only care about scalar type values, which is currently limited to:
      // felt, index, etc.

      if (!mlir::isa<FeltType>(val.getType())) {
        operandVals.push_back(LatticeValue());
        continue;
      }
      auto refSet = constrainRefLattice->getOrDefault(val);
      // todo fix single value
      auto ref = refSet.getSingleValue();
      auto exprVal = ExpressionValue(getOrCreateSymbol(ref));
      operandVals.emplace_back(exprVal);
    }

    // Now, the way we update is dependent on the type of the operation.
    if (!isConsideredOp(op)) {
      llvm::errs() << *op << " is unconsidered!\n";
      ensure(isConsideredOp(op), "unconsidered op");
    }

    mlir::ChangeResult changed = after->join(before);
    if (isConstOp(op)) {
      auto constVal = getConst(op);
      auto resVal = op->getResult(0);
      llvm::errs() << "gonna set " << resVal << " to " << constVal << "\n";
      auto expr = createFeltSymbol(resVal);
      auto latticeVal = ExpressionValue(expr, constVal);
      llvm::errs() << "setting to " << latticeVal << '\n';
      changed |= after->setValue(resVal, latticeVal);
    } else if (isArithmeticOp(op)) {
      ensure(operandVals.size() == 2, "arithmetic op with the wrong number of operands");
      auto result = performArithmetic(op, operandVals[0], operandVals[1]);
      llvm::errs() << "op " << *op << " result is " << result << "\n";
      changed |= after->setValue(op->getResult(0), result);
    } else if (isReturnOp(op)) {
      llvm::errs() << "TODO return?\n";
    } else if (isEmitOp(op)) {
      ensure(operandVals.size() == 2, "constraint op with the wrong number of operands");
      if (mlir::isa<EmitContainmentOp>(op)) {
        llvm::report_fatal_error("todo, not impl");
      }
      auto lhsVal = op->getOperand(0);
      auto rhsVal = op->getOperand(1);
      auto lhsExpr = operandVals[0].getScalarValue();
      auto rhsExpr = operandVals[1].getScalarValue();

      // TODO, revert this so we update the lattice for lhs and rhs with new intervals.
      auto constraint = intersection(smtSolver, lhsExpr, rhsExpr);
      // Update the LHS and RHS to the same value, but restricted intervals
      // based on the constraints
      changed |=
          after->setValue(lhsVal, ExpressionValue(lhsExpr.getExpr(), constraint.getInterval()));
      changed |=
          after->setValue(rhsVal, ExpressionValue(rhsExpr.getExpr(), constraint.getInterval()));
      changed |= after->addSolverConstraint(constraint);

      llvm::errs() << "emit: " << constraint << "\n";

    } else if (!isIgnoredOp(op) && !isDefinitionOp(op)) {
      llvm::errs() << "TODO: " << *op << "\n";
      llvm::report_fatal_error("unimplemented else branch");
    }

    llvm::errs() << "after lattice: " << *after << "\n";

    propagateIfChanged(after, changed);
  }

  llvm::SMTExprRef getOrCreateSymbol(const ConstrainRef &r) {
    auto it = refSymbols.find(r);
    if (it != refSymbols.end()) {
      return it->second;
    }
    auto sym = createFeltSymbol(r);
    refSymbols[r] = sym;
    return sym;
  }

private:
  mlir::DataFlowSolver &dataflowSolver;
  llvm::SMTSolverRef smtSolver;
  SymbolMap refSymbols;

  void setToEntryState(Lattice *lattice) override {
    // initial state should be empty, so do nothing here
  }

  llvm::SMTExprRef createFeltSymbol(const ConstrainRef &r) const {
    std::string symbolName;
    llvm::raw_string_ostream ss(symbolName);
    r.print(ss);

    return createFeltSymbol(symbolName.c_str());
  }

  llvm::SMTExprRef createFeltSymbol(mlir::Value val) const {
    std::string symbolName;
    llvm::raw_string_ostream ss(symbolName);
    val.print(ss);

    // todo: bitvector size
    return createFeltSymbol(symbolName.c_str());
  }

  llvm::SMTExprRef createFeltSymbol(const char *name) const {
    // todo: bitvector size
    return smtSolver->mkSymbol(name, smtSolver->getBitvectorSort(254));
  }

  bool isConstOp(mlir::Operation *op) const { return mlir::isa<FeltConstantOp>(op); }

  mlir::APInt getConst(mlir::Operation *op) const {
    ensure(isConstOp(op), "op is not a const op");
    return mlir::dyn_cast<FeltConstantOp>(op).getValueAttr().getValue();
  }

  bool isArithmeticOp(mlir::Operation *op) const {
    return isOneOf<
        AddFeltOp, SubFeltOp, MulFeltOp, DivFeltOp, ModFeltOp, NegFeltOp, InvFeltOp, AndFeltOp,
        OrFeltOp, XorFeltOp, NotFeltOp, ShlFeltOp, ShrFeltOp, CmpOp>(op);
  }

  ExpressionValue
  performArithmetic(mlir::Operation *op, const LatticeValue &a, const LatticeValue &b) {
    ensure(isArithmeticOp(op), "is not arithmetic op");

    auto lhs = a.getScalarValue(), rhs = b.getScalarValue();

    if (mlir::isa<AddFeltOp>(op)) {
      return add(smtSolver, lhs, rhs);
    } else {
      llvm::errs() << __FUNCTION__ << ": unhandled op " << *op << "\n";
      llvm::report_fatal_error("oops");
    }
    return ExpressionValue();
  }

  bool isBoolOp(mlir::Operation *op) const {
    return isOneOf<AndBoolOp, OrBoolOp, XorBoolOp, NotBoolOp>(op);
  }

  bool isConversionOp(mlir::Operation *op) const { return isOneOf<IntToFeltOp, FeltToIndexOp>(op); }

  bool isApplyMapOp(mlir::Operation *op) const { return mlir::isa<ApplyMapOp>(op); }

  bool isAssertOp(mlir::Operation *op) const { return mlir::isa<AssertOp>(op); }

  bool isReadOp(mlir::Operation *op) const {
    return isOneOf<FieldReadOp, ConstReadOp, ReadArrayOp>(op);
  }

  bool isWriteOp(mlir::Operation *op) const {
    return isOneOf<FieldWriteOp, WriteArrayOp, InsertArrayOp>(op);
  }

  bool isArrayLengthOp(mlir::Operation *op) const { return mlir::isa<ArrayLengthOp>(op); }

  bool isEmitOp(mlir::Operation *op) const {
    return isOneOf<EmitEqualityOp, EmitContainmentOp>(op);
  }

  bool isIgnoredOp(mlir::Operation *op) const {
    return isOneOf<CreateStructOp, CreateArrayOp, ExtractArrayOp>(op);
  }

  bool isDefinitionOp(mlir::Operation *op) const { return isOneOf<StructDefOp, FuncOp>(op); }

  bool isCallOp(mlir::Operation *op) const { return mlir::isa<CallOp>(op); }

  bool isReturnOp(mlir::Operation *op) const { return mlir::isa<ReturnOp>(op); }

  bool isConsideredOp(mlir::Operation *op) const {
    return isConstOp(op) || isArithmeticOp(op) || isBoolOp(op) || isConversionOp(op) ||
           isApplyMapOp(op) || isAssertOp(op) || isReadOp(op) || isWriteOp(op) ||
           isArrayLengthOp(op) || isEmitOp(op) || isIgnoredOp(op) || isDefinitionOp(op) ||
           isCallOp(op) || isReturnOp(op);
  }
};

/* StructIntervals */

struct IntervalAnalysisContext {
  IntervalDataFlowAnalysis *intervalDFA;
  llvm::SMTSolverRef smtSolver;

  llvm::SMTExprRef getSymbol(const ConstrainRef &r) { return intervalDFA->getOrCreateSymbol(r); }
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
      mlir::ModuleOp mod, StructDefOp s, mlir::DataFlowSolver &solver, mlir::AnalysisManager &am,
      IntervalAnalysisContext &ctx
  ) {
    StructIntervals si(mod, s);
    if (si.computeIntervals(solver, am, ctx).failed()) {
      return mlir::failure();
    }
    return si;
  }

  mlir::LogicalResult computeIntervals(
      mlir::DataFlowSolver &solver, mlir::AnalysisManager &am, IntervalAnalysisContext &ctx
  );

  void print(mlir::raw_ostream &os) const;

  friend mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const StructIntervals &si) {
    si.print(os);
    return os;
  }

private:
  mlir::ModuleOp mod;
  StructDefOp structDef;
  llvm::SMTSolverRef smtSolver;
  mlir::DenseMap<ConstrainRef, Interval> computeFieldRanges, constrainFieldRanges;
  mlir::DenseSet<ExpressionValue> computeSolverConstraints, constrainSolverConstraints;

  StructIntervals(mlir::ModuleOp m, StructDefOp s) : mod(m), structDef(s) {}
};

/* StructIntervalAnalysis */

class ModuleIntervalAnalysis;

class StructIntervalAnalysis : public StructAnalysis<StructIntervals, IntervalAnalysisContext> {
public:
  using StructAnalysis::StructAnalysis;

  mlir::LogicalResult runAnalysis(
      mlir::DataFlowSolver &solver, mlir::AnalysisManager &moduleAnalysisManager,
      IntervalAnalysisContext &ctx
  ) override {
    auto res =
        StructIntervals::compute(getModule(), getStruct(), solver, moduleAnalysisManager, ctx);
    if (mlir::failed(res)) {
      return mlir::failure();
    }
    setResult(std::move(*res));
    return mlir::success();
  }
};

/* ModuleIntervalAnalysis */

class ModuleIntervalAnalysis
    : public ModuleAnalysis<StructIntervals, IntervalAnalysisContext, StructIntervalAnalysis> {

public:
  ModuleIntervalAnalysis(mlir::Operation *op, mlir::AnalysisManager &am)
      : ModuleAnalysis(op, am), smtSolver(llvm::CreateZ3Solver()) {
    constructChildAnalyses(am);
  }

protected:
  void initializeSolver(mlir::DataFlowSolver &solver) override {
    (void)solver.load<ConstrainRefAnalysis>();
    auto smtSolverRef = smtSolver;
    intervalDFA =
        solver.load<IntervalDataFlowAnalysis, llvm::SMTSolverRef>(std::move(smtSolverRef));
  }

  IntervalAnalysisContext getContext() override {
    return {
        .intervalDFA = intervalDFA,
        .smtSolver = smtSolver,
    };
  }

private:
  llvm::SMTSolverRef smtSolver;
  IntervalDataFlowAnalysis *intervalDFA;
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

  static llzk::ExpressionValue getEmptyKey() { return llzk::ExpressionValue(getEmptyExpr()); }
  static inline llzk::ExpressionValue getTombstoneKey() {
    return llzk::ExpressionValue(getTombstoneExpr());
  }
  static unsigned getHashValue(const llzk::ExpressionValue &e) {
    return llzk::ExpressionValue::Hash {}(e);
  }
  static bool isEqual(const llzk::ExpressionValue &lhs, const llzk::ExpressionValue &rhs) {
    if (lhs.getExpr() == getEmptyExpr() || lhs.getExpr() == getTombstoneExpr()) {
      return lhs.getExpr() == rhs.getExpr();
    }
    return lhs == rhs;
  }
};

} // namespace llvm
