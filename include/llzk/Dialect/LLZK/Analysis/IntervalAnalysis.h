//===-- IntervalAnalysis.h --------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/LLZK/Analysis/AbstractLatticeValue.h"
#include "llzk/Dialect/LLZK/Analysis/AnalysisWrappers.h"
#include "llzk/Dialect/LLZK/Analysis/ConstraintDependencyGraph.h"
#include "llzk/Dialect/LLZK/Analysis/DenseAnalysis.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/Util/APIntHelper.h"
#include "llzk/Dialect/LLZK/Util/Compare.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/AnalysisManager.h>
#include <mlir/Support/LLVM.h>

#include <llvm/Support/SMTAPI.h>

#include <array>
#include <mutex>

#include "llvm/ADT/MapVector.h"

namespace llzk {

/* Field */

/// @brief Information about the prime finite field used for the interval analysis.
/// @note Seem implementation of initKnownFields for supported primes.
class Field {
public:
  /// @brief Get a Field from a given field name string.
  /// @param fieldName The name of the field.
  static const Field &getField(const char *fieldName);

  Field() = delete;
  Field(const Field &) = default;
  Field(Field &&) = default;
  Field &operator=(const Field &) = default;

  /// @brief For the prime field p, returns p.
  llvm::APSInt prime() const { return primeMod; }

  /// @brief Returns p / 2.
  llvm::APSInt half() const { return halfPrime; }

  /// @brief Returns i as a field element
  inline llvm::APSInt felt(unsigned i) const { return reduce(i); }

  /// @brief Returns 0 at the bitwidth of the field.
  inline llvm::APSInt zero() const { return felt(0); }

  /// @brief Returns 1 at the bitwidth of the field.
  inline llvm::APSInt one() const { return felt(1); }

  /// @brief Returns p - 1, which is the max value possible in a prime field described by p.
  inline llvm::APSInt maxVal() const { return prime() - one(); }

  /// @brief Returns i mod p and reduces the result into the appropriate bitwidth.
  llvm::APSInt reduce(llvm::APSInt i) const;
  llvm::APSInt reduce(unsigned i) const;

  inline unsigned bitWidth() const { return primeMod.getBitWidth(); }

  /// @brief Create a SMT solver symbol with the current field's bitwidth.
  llvm::SMTExprRef createSymbol(llvm::SMTSolverRef solver, const char *name) const {
    return solver->mkSymbol(name, solver->getBitvectorSort(bitWidth()));
  }

  friend bool operator==(const Field &lhs, const Field &rhs) {
    return lhs.primeMod == rhs.primeMod;
  }

private:
  Field(std::string_view primeStr);
  Field(llvm::APSInt p, llvm::APSInt h) : primeMod(p), halfPrime(h) {}

  llvm::APSInt primeMod, halfPrime;

  static void initKnownFields(llvm::DenseMap<llvm::StringRef, Field> &knownFields);
};

/* UnreducedInterval */

class Interval;

/// @brief An inclusive interval [a, b] where a and b are arbitrary integers
/// not necessarily bound to a given field.
class UnreducedInterval {
public:
  /// @brief A utility method to determine the largest bitwidth among arms of two
  /// UnreducedIntervals. Useful for widening integers for comparisons between APInts.
  /// TODO: When we upgrade to LLVM 19/20, we can instead use DynamicAPInts to avoid
  /// the messy widening/narrowing logic.
  /// @param lhs
  /// @param rhs
  /// @return
  static size_t getMaxBitWidth(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
    return std::max(
        {lhs.a.getBitWidth(), lhs.b.getBitWidth(), rhs.a.getBitWidth(), rhs.b.getBitWidth()}
    );
  }

  UnreducedInterval(llvm::APSInt x, llvm::APSInt y) : a(x), b(y) {}
  UnreducedInterval(llvm::APInt x, llvm::APInt y) : a(x), b(y) {}
  /// @brief This constructor is primarily for convenience for unit tests.
  UnreducedInterval(uint64_t x, uint64_t y) : a(llvm::APInt(64, x)), b(llvm::APInt(64, y)) {}

  /* Operations */

  /// @brief Reduce the interval to an interval in the given field.
  /// @param field
  /// @return
  Interval reduce(const Field &field) const;

  /// @brief Compute and return the intersection of this interval and the given RHS.
  /// @param rhs
  /// @return
  UnreducedInterval intersect(const UnreducedInterval &rhs) const;

  /// @brief Compute and return the union of this interval and the given RHS.
  /// @param rhs
  /// @return
  UnreducedInterval doUnion(const UnreducedInterval &rhs) const;

  /// @brief Return the part of the interval that is guaranteed to be less than
  /// the rhs's max value.
  ///
  /// For example, given *this = [0, 7] and rhs = [3, 5], this function would
  /// return [0, 4], since rhs has a max value of 5. If this interval's lower
  /// bound is greater than or equal to the rhs's upper bound, the returned
  /// interval will be "empty" (an interval where a > b). For example,
  /// if *this = [7, 10] and rhs = [0, 7], then no part of *this is less than rhs.
  UnreducedInterval computeLTPart(const UnreducedInterval &rhs) const;

  /// @brief Return the part of the interval that is less than or equal to the
  /// rhs's upper bound.
  ///
  /// For example, given *this = [0, 7] and rhs = [3, 5], this function would
  /// return [0, 5], since rhs has a max value of 5. If this interval's lower
  /// bound is greater than to the rhs's upper bound, the returned
  /// interval will be "empty" (an interval where a > b). For example, if
  /// *this = [8, 10] and rhs = [0, 7], then no part of *this is less than or equal to rhs.
  UnreducedInterval computeLEPart(const UnreducedInterval &rhs) const;

  /// @brief Return the part of the interval that is greater than the rhs's
  /// lower bound.
  ///
  /// For example, given *this = [0, 7] and rhs = [3, 5], this function would
  /// return [4, 7], since rhs has a minimum value of 3. If this interval's
  /// upper bound is less than or equal to the rhs's lower bound, the returned
  /// interval will be "empty" (an interval where a > b). For example,
  /// if *this = [0, 7] and rhs = [7, 10], then no part of *this is greater than rhs.
  UnreducedInterval computeGTPart(const UnreducedInterval &rhs) const;

  /// @brief Return the part of the interval that is greater than or equal to
  /// the rhs's lower bound.
  ///
  /// For example, given *this = [0, 7] and rhs = [3, 5], this function would
  /// return [3, 7], since rhs has a minimum value of 3. If this interval's
  /// upper bound is less than the rhs's lower bound, the returned
  /// interval will be "empty" (an interval where a > b). For example, if
  /// *this = [0, 6] and rhs = [7, 10], then no part of *this is greater than or equal to rhs.
  UnreducedInterval computeGEPart(const UnreducedInterval &rhs) const;

  UnreducedInterval operator-() const;
  friend UnreducedInterval operator+(const UnreducedInterval &lhs, const UnreducedInterval &rhs);
  friend UnreducedInterval operator-(const UnreducedInterval &lhs, const UnreducedInterval &rhs);
  friend UnreducedInterval operator*(const UnreducedInterval &lhs, const UnreducedInterval &rhs);

  /* Comparisons */

  bool overlaps(const UnreducedInterval &rhs) const;

  friend std::strong_ordering
  operator<=>(const UnreducedInterval &lhs, const UnreducedInterval &rhs);

  friend bool operator==(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
    return std::is_eq(lhs <=> rhs);
  };

  /* Utility */
  llvm::APSInt getLHS() const { return a; }
  llvm::APSInt getRHS() const { return b; }

  /// @brief Compute the width of this interval within a given field `f`.
  /// If `a` > `b`, returns 0. Otherwise, returns `b` - `a` + 1.
  llvm::APSInt width() const;

  /// @brief Returns true iff width() is zero.
  bool isEmpty() const;

  bool isNotEmpty() const { return !isEmpty(); }

  void print(llvm::raw_ostream &os) const { os << "Unreduced:[ " << a << ", " << b << " ]"; }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const UnreducedInterval &ui) {
    ui.print(os);
    return os;
  }

private:
  llvm::APSInt a, b;
};

/* Interval */

/// @brief Intervals over a finite field. Based on the Picus implementation.
/// An interval may be:
/// - Empty
/// - Entire (meaning any value across the entire field)
/// - Degenerate (meaning it contains a single value)
/// - Or Type A--F. For these types, refer to the below notes:
///
/// A range [a, b] can be split into 2 categories:
/// - Internal: a <= b
/// - External: a > b -- equivalent to [a, p-1] U [0, b]
///
/// Internal range can be further split into 3 categories:
/// (A) a, b < p/2.                                             E.g., [10, 12]
/// (B) a, b > p/2.       OR: a, b \in {-p/2, 0}.               E.g., [p-4, p-2] === [-4, -2]
/// (C) a < p/2, b > p/2.                                       E.g., [p/2 - 5, p/2 + 5]
///
/// External range can be further split into 3 categories:
/// (D) a, b < p/2.       OR: a \in {-p, -p/2}, b \in {0, p/2}. E.g., [12, 10] === [-p+12, 10]
/// (E) a, b > p/2.       OR: a \in {-p/2, 0} , b \in {p/2, p}. E.g., [p-2, p-4] === [-2, p-4]
/// (F) a > p/2, b < p/2. OR: a \in {-p/2, 0} , b \in {0, p/2}. E.g., [p/2 + 5, p/2 - 5]
/// === [-p/2 + 5, p/2 - 5]
///
/// <------------------------------------------------------------->
///   -p           -p/2            0            p/2             p
///      [  A  ]                      [  A  ]
///                       [  B  ]                     [  B  ]
///             [    C    ]                 [    C    ]
///     F     ]              [       F       ]           [      F
/// <------------------------------------------------------------->
///
///   D      ]  [           D            ]  [           D            ]
///          E            ]  [            E              ]  [   E
///
/// For the sake of simplicity, let's just not care about D and E, which covers at least
/// half of the field, and potentially more.
///
/// Now, there are 4 choose 2 possible non-self interactions:
///
/// A acts on B:
/// - intersection: impossible
/// - union: C or F
///
/// A acts on C:
/// - intersection: A
/// - union: C
///
/// A acts on F:
/// - intersection: A
/// - union: F
///
/// B acts on C
/// - intersection: B
/// - union: C
///
/// B acts on F:
/// - intersection: B
/// - union: F
///
/// C acts on F:
/// - intersection: A, B, C, F
///
///   E.g. [p/2 - 10, p/2 + 10] intersects [-p/2 + 2, p/2 - 2]
///
///   = ((-p/2 - 10, -p/2 + 10) intersects (-p/2 + 2, p/2 - 2)) union
///     (( p/2 - 10,  p/2 + 10) intersects (-p/2 + 2, p/2 - 2))
///
///   = (-p/2 + 2, -p/2 + 10) union (p/2 - 10, p/2 - 2)
///
/// - union: don't care for now, we can revisit this later.
class Interval {
public:
  enum class Type { TypeA = 0, TypeB, TypeC, TypeF, Empty, Degenerate, Entire };
  static constexpr std::array<std::string_view, 7> TypeNames = {"TypeA", "TypeB", "TypeC",
                                                                "TypeF", "Empty", "Degenerate",
                                                                "Entire"};

  static std::string_view TypeName(Type t) { return TypeNames.at(static_cast<size_t>(t)); }

  /* Static constructors for convenience */

  static Interval Empty(const Field &f) { return Interval(Type::Empty, f); }

  static Interval Degenerate(const Field &f, llvm::APSInt val) {
    return Interval(Type::Degenerate, f, val, val);
  }

  static Interval Boolean(const Field &f) { return Interval::TypeA(f, f.zero(), f.one()); }

  static Interval Entire(const Field &f) { return Interval(Type::Entire, f); }

  static Interval TypeA(const Field &f, llvm::APSInt a, llvm::APSInt b) {
    return Interval(Type::TypeA, f, a, b);
  }

  static Interval TypeB(const Field &f, llvm::APSInt a, llvm::APSInt b) {
    return Interval(Type::TypeB, f, a, b);
  }

  static Interval TypeC(const Field &f, llvm::APSInt a, llvm::APSInt b) {
    return Interval(Type::TypeC, f, a, b);
  }

  static Interval TypeF(const Field &f, llvm::APSInt a, llvm::APSInt b) {
    return Interval(Type::TypeF, f, a, b);
  }

  /// To satisfy the dataflow::ScalarLatticeValue requirements, this class must
  /// be default initializable. The default interval is the full range of values.
  Interval() : Interval(Type::Entire, Field::getField("bn128")) {}

  /// @brief Convert to an UnreducedInterval.
  UnreducedInterval toUnreduced() const;

  /// @brief Get the first side of the interval for TypeF intervals, otherwise
  /// just get the full interval as an UnreducedInterval (with toUnreduced).
  UnreducedInterval firstUnreduced() const;

  /// @brief Get the second side of the interval for TypeA, TypeB, and TypeC intervals.
  /// Using this function is an error for all other interval types.
  UnreducedInterval secondUnreduced() const;

  template <std::pair<Type, Type>... Pairs>
  static bool areOneOf(const Interval &a, const Interval &b) {
    return ((a.ty == std::get<0>(Pairs) && b.ty == std::get<1>(Pairs)) || ...);
  }

  /// Union
  Interval join(const Interval &rhs) const;

  /// Intersect
  Interval intersect(const Interval &rhs) const;

  /// @brief Computes and returns `this` - (`this` & `other`) if the operation
  /// produces a single interval.
  ///
  /// Note that this is an interval difference, not a subtraction operation
  /// like the `operator-` below.
  ///
  /// For example, given `*this` = [1, 10] and `other` = [5, 11], this function
  /// would return [1, 4], as `this` & `other` (the intersection) = [5, 10], so
  /// [1, 10] - [5, 10] = [1, 4].
  ///
  /// For example, given `*this` = [1, 10] and `other` = [5, 6], this function
  /// should return [1, 4] and [7, 10], but we don't support having multiple
  /// disjoint intervals, so `this` is returned as-is.
  Interval difference(const Interval &other) const;

  /* arithmetic ops */

  Interval operator-() const;
  friend Interval operator+(const Interval &lhs, const Interval &rhs);
  friend Interval operator-(const Interval &lhs, const Interval &rhs);
  friend Interval operator*(const Interval &lhs, const Interval &rhs);
  friend Interval operator%(const Interval &lhs, const Interval &rhs);
  /// @brief Returns failure if a division-by-zero is encountered.
  friend mlir::FailureOr<Interval> operator/(const Interval &lhs, const Interval &rhs);

  /* Checks and Comparisons */

  inline bool isEmpty() const { return ty == Type::Empty; }
  inline bool isNotEmpty() const { return !isEmpty(); }
  inline bool isDegenerate() const { return ty == Type::Degenerate; }
  inline bool isEntire() const { return ty == Type::Entire; }
  inline bool isTypeA() const { return ty == Type::TypeA; }
  inline bool isTypeB() const { return ty == Type::TypeB; }
  inline bool isTypeC() const { return ty == Type::TypeC; }
  inline bool isTypeF() const { return ty == Type::TypeF; }

  template <Type... Types> bool is() const { return ((ty == Types) || ...); }

  bool operator==(const Interval &rhs) const { return ty == rhs.ty && a == rhs.a && b == rhs.b; }

  /* Getters */

  const Field &getField() const { return field.get(); }

  llvm::APSInt width() const { return llvm::APSInt((b - a).abs().zext(field.get().bitWidth())); }

  llvm::APSInt lhs() const { return a; }
  llvm::APSInt rhs() const { return b; }

  /* Utility */
  struct Hash {
    unsigned operator()(const Interval &i) const {
      return std::hash<const Field *> {}(&i.field.get()) ^ std::hash<Type> {}(i.ty) ^
             llvm::hash_value(i.a) ^ llvm::hash_value(i.b);
    }
  };

  void print(mlir::raw_ostream &os) const;

  friend mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const Interval &i) {
    i.print(os);
    return os;
  }

private:
  Interval(Type t, const Field &f) : field(f), ty(t), a(f.zero()), b(f.zero()) {}
  Interval(Type t, const Field &f, llvm::APSInt lhs, llvm::APSInt rhs)
      : field(f), ty(t), a(lhs.extend(f.bitWidth())), b(rhs.extend(f.bitWidth())) {}

  std::reference_wrapper<const Field> field;
  Type ty;
  llvm::APSInt a, b;
};

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
  div(llvm::SMTSolverRef solver, DivFeltOp op, const ExpressionValue &lhs,
      const ExpressionValue &rhs);

  friend ExpressionValue
  mod(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs);

  friend ExpressionValue
  cmp(llvm::SMTSolverRef solver, CmpOp op, const ExpressionValue &lhs, const ExpressionValue &rhs);

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

  mlir::ChangeResult setValue(mlir::Value v, ExpressionValue e);

  mlir::ChangeResult addSolverConstraint(ExpressionValue e);

  friend mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const IntervalAnalysisLattice &l) {
    l.print(os);
    return os;
  }

  const ConstraintSet &getConstraints() const { return constraints; }

  mlir::FailureOr<Interval> findInterval(llvm::SMTExprRef expr) const;

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
  explicit IntervalDataFlowAnalysis(
      mlir::DataFlowSolver &solver, llvm::SMTSolverRef smt, const Field &f
  )
      : Base::DenseForwardDataFlowAnalysis(solver), dataflowSolver(solver), smtSolver(smt),
        field(f) {}

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
  mlir::DataFlowSolver &dataflowSolver;
  llvm::SMTSolverRef smtSolver;
  SymbolMap refSymbols;
  std::reference_wrapper<const Field> field;

  void setToEntryState(Lattice *lattice) override {
    // initial state should be empty, so do nothing here
  }

  llvm::SMTExprRef createFeltSymbol(const ConstrainRef &r) const;

  llvm::SMTExprRef createFeltSymbol(mlir::Value val) const;

  llvm::SMTExprRef createFeltSymbol(const char *name) const;

  bool isConstOp(mlir::Operation *op) const {
    return mlir::isa<FeltConstantOp, mlir::arith::ConstantIndexOp, mlir::arith::ConstantIntOp>(op);
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
        AddFeltOp, SubFeltOp, MulFeltOp, DivFeltOp, ModFeltOp, NegFeltOp, InvFeltOp, AndFeltOp,
        OrFeltOp, XorFeltOp, NotFeltOp, ShlFeltOp, ShrFeltOp, CmpOp>(op);
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

  bool isBoolOp(mlir::Operation *op) const {
    return mlir::isa<AndBoolOp, OrBoolOp, XorBoolOp, NotBoolOp>(op);
  }

  bool isConversionOp(mlir::Operation *op) const {
    return mlir::isa<IntToFeltOp, FeltToIndexOp>(op);
  }

  bool isApplyMapOp(mlir::Operation *op) const { return mlir::isa<ApplyMapOp>(op); }

  bool isAssertOp(mlir::Operation *op) const { return mlir::isa<AssertOp>(op); }

  bool isReadOp(mlir::Operation *op) const {
    return mlir::isa<FieldReadOp, ConstReadOp, ReadArrayOp>(op);
  }

  bool isWriteOp(mlir::Operation *op) const {
    return mlir::isa<FieldWriteOp, WriteArrayOp, InsertArrayOp>(op);
  }

  bool isArrayLengthOp(mlir::Operation *op) const { return mlir::isa<ArrayLengthOp>(op); }

  bool isEmitOp(mlir::Operation *op) const {
    return mlir::isa<EmitEqualityOp, EmitContainmentOp>(op);
  }

  bool isCreateOp(mlir::Operation *op) const {
    return mlir::isa<CreateStructOp, CreateArrayOp>(op);
  }

  bool isExtractArrayOp(mlir::Operation *op) const { return mlir::isa<ExtractArrayOp>(op); }

  bool isDefinitionOp(mlir::Operation *op) const {
    return mlir::isa<StructDefOp, FuncOp, FieldDefOp, GlobalDefOp, mlir::ModuleOp>(op);
  }

  bool isCallOp(mlir::Operation *op) const { return mlir::isa<CallOp>(op); }

  bool isReturnOp(mlir::Operation *op) const { return mlir::isa<ReturnOp>(op); }

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
  Field field;

  llvm::SMTExprRef getSymbol(const ConstrainRef &r) { return intervalDFA->getOrCreateSymbol(r); }
  const Field &getField() const { return field; }
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

  void print(mlir::raw_ostream &os, bool withConstraints = false) const;

  const llvm::MapVector<ConstrainRef, Interval> &getIntervals() const {
    return constrainFieldRanges;
  }

  const llvm::SetVector<ExpressionValue> getSolverConstraints() const {
    return constrainSolverConstraints;
  }

  friend mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const StructIntervals &si) {
    si.print(os);
    return os;
  }

private:
  mlir::ModuleOp mod;
  StructDefOp structDef;
  llvm::SMTSolverRef smtSolver;
  // llvm::MapVector keeps insertion order for consistent iteration
  llvm::MapVector<ConstrainRef, Interval> constrainFieldRanges;
  // llvm::SetVector for the same reasons as above
  llvm::SetVector<ExpressionValue> constrainSolverConstraints;

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
  ModuleIntervalAnalysis(mlir::Operation *op)
      : ModuleAnalysis(op), smtSolver(llvm::CreateZ3Solver()), field(std::nullopt) {}

  void setField(const Field &f) { field = f; }

protected:
  void initializeSolver(mlir::DataFlowSolver &solver) override {
    ensure(field.has_value(), "field not set, could not generate analysis context");
    (void)solver.load<ConstrainRefAnalysis>();
    auto smtSolverRef = smtSolver;
    intervalDFA = solver.load<IntervalDataFlowAnalysis, llvm::SMTSolverRef, const Field &>(
        std::move(smtSolverRef), field.value()
    );
  }

  IntervalAnalysisContext getContext() override {
    ensure(field.has_value(), "field not set, could not generate analysis context");
    return {
        .intervalDFA = intervalDFA,
        .smtSolver = smtSolver,
        .field = field.value(),
    };
  }

private:
  llvm::SMTSolverRef smtSolver;
  IntervalDataFlowAnalysis *intervalDFA;
  std::optional<Field> field;
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
