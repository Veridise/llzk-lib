#include "llzk/Dialect/LLZK/Analysis/IntervalAnalysis.h"
#include "llzk/Dialect/LLZK/Util/Debug.h"

#include <llvm/ADT/TypeSwitch.h>

using namespace mlir;

namespace llzk {

/* Field */

Field::Field(std::string_view primeStr) : primeMod(llvm::APSInt(primeStr)) {
  halfPrime = (primeMod + felt(1)) / felt(2);
}

const Field &Field::getField(const char *fieldName) {
  static llvm::DenseMap<llvm::StringRef, Field> knownFields;
  static std::once_flag fieldsInit;
  std::call_once(fieldsInit, initKnownFields, knownFields);

  if (auto it = knownFields.find(fieldName); it != knownFields.end()) {
    return it->second;
  }
  llvm::report_fatal_error("field \"" + mlir::Twine(fieldName) + "\" is unsupported");
}

void Field::initKnownFields(llvm::DenseMap<llvm::StringRef, Field> &knownFields) {
  // bn128/254, default for circom
  knownFields.try_emplace(
      "bn128",
      Field("21888242871839275222246405745257275088696311157297823662689037894645226208583")
  );
  knownFields.try_emplace("bn254", knownFields.at("bn128"));
  // 15 * 2^27 + 1, default for zirgen
  knownFields.try_emplace("babybear", Field("2013265921"));
  // 2^64 - 2^32 + 1, used for plonky2
  knownFields.try_emplace("goldilocks", Field("18446744069414584321"));
  // 2^31 - 1, used for Plonky3
  knownFields.try_emplace("mersenne31", Field("2147483647"));
}

llvm::APSInt Field::reduce(llvm::APSInt i) const {
  auto maxBits = std::max(i.getBitWidth(), bitWidth());
  auto m = (i.zext(maxBits).urem(prime().zext(maxBits))).trunc(bitWidth());
  if (m.isNegative()) {
    return prime() + llvm::APSInt(m);
  }
  return llvm::APSInt(m);
}

llvm::APSInt Field::reduce(unsigned i) const {
  auto ap = llvm::APSInt(llvm::APInt(bitWidth(), i));
  return reduce(ap);
}

/* UnreducedInterval */

Interval UnreducedInterval::reduce(const Field &field) const {
  if (a > b) {
    return Interval::Empty(field);
  }
  if (width().trunc(field.bitWidth()) >= field.prime()) {
    return Interval::Entire(field);
  }

  auto lhs = field.reduce(a), rhs = field.reduce(b);

  if ((rhs - lhs).isZero()) {
    return Interval::Degenerate(field, lhs);
  }

  const auto &half = field.half();
  if (lhs.ule(rhs)) {
    if (lhs.ult(half) && rhs.ult(half)) {
      return Interval::TypeA(field, lhs, rhs);
    } else if (lhs.ult(half)) {
      return Interval::TypeC(field, lhs, rhs);
    } else {
      return Interval::TypeB(field, lhs, rhs);
    }
  } else {
    if (lhs.uge(half) && rhs.ult(half)) {
      return Interval::TypeF(field, lhs, rhs);
    } else {
      return Interval::Entire(field);
    }
  }
}

UnreducedInterval UnreducedInterval::intersect(const UnreducedInterval &rhs) const {
  auto &lhs = *this;
  return UnreducedInterval(std::max(lhs.a, rhs.a), std::min(lhs.b, rhs.b));
}

UnreducedInterval UnreducedInterval::doUnion(const UnreducedInterval &rhs) const {
  auto &lhs = *this;
  return UnreducedInterval(std::min(lhs.a, rhs.a), std::max(lhs.b, rhs.b));
}

UnreducedInterval UnreducedInterval::lt(const UnreducedInterval &rhs) const {
  auto one = llvm::APSInt(llvm::APInt(a.getBitWidth(), 1));
  auto bound = rhs.b - one;
  return UnreducedInterval(std::min(a, bound), std::min(b, bound));
}

UnreducedInterval UnreducedInterval::le(const UnreducedInterval &rhs) const {
  return UnreducedInterval(std::min(a, rhs.b), std::min(b, rhs.b));
}

UnreducedInterval UnreducedInterval::gt(const UnreducedInterval &rhs) const {
  auto one = llvm::APSInt(llvm::APInt(a.getBitWidth(), 1));
  auto bound = rhs.a + one;
  return UnreducedInterval(std::max(a, bound), std::max(b, bound));
}

UnreducedInterval UnreducedInterval::ge(const UnreducedInterval &rhs) const {
  return UnreducedInterval(std::max(a, rhs.a), std::max(b, rhs.a));
}

UnreducedInterval UnreducedInterval::operator-() const { return UnreducedInterval(-b, -a); }

UnreducedInterval operator+(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
  return UnreducedInterval(lhs.a + rhs.a, lhs.b + rhs.b);
}

UnreducedInterval operator-(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
  return lhs + (-rhs);
}

UnreducedInterval operator*(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
  auto v1 = lhs.a * rhs.a;
  auto v2 = lhs.a * rhs.b;
  auto v3 = lhs.b * rhs.a;
  auto v4 = lhs.b * rhs.b;

  auto minVal = std::min({v1, v2, v3, v4});
  auto maxVal = std::max({v1, v2, v3, v4});

  return UnreducedInterval(minVal, maxVal);
}

bool UnreducedInterval::overlaps(const UnreducedInterval &rhs) const {
  auto &lhs = *this;
  return lhs.b >= rhs.a || lhs.a <= rhs.b;
}

std::strong_ordering operator<=>(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
  if (lhs.a < rhs.a || (lhs.a == rhs.a && lhs.b < rhs.b)) {
    return std::strong_ordering::less;
  }
  if (lhs.a > rhs.a || (lhs.a == rhs.a && lhs.b > rhs.b)) {
    return std::strong_ordering::greater;
  }
  return std::strong_ordering::equal;
}

/* Interval */

UnreducedInterval Interval::toUnreduced() const {
  if (isEmpty()) {
    return UnreducedInterval(field.get().zero(), field.get().zero());
  }
  if (isEntire()) {
    return UnreducedInterval(field.get().zero(), field.get().maxVal());
  }
  return UnreducedInterval(a, b);
}

UnreducedInterval Interval::firstUnreduced() const {
  if (is<Type::TypeF>()) {
    return UnreducedInterval(field.get().prime() - a, b);
  }
  return toUnreduced();
}

UnreducedInterval Interval::secondUnreduced() const {
  ensure(is<Type::TypeA, Type::TypeB, Type::TypeC>(), "unsupported range type");
  return UnreducedInterval(a - field.get().prime(), b - field.get().prime());
}

Interval Interval::join(const Interval &rhs) const {
  auto &lhs = *this;

  // Trivial cases
  if (lhs.isEntire() || rhs.isEntire()) {
    return Interval::Entire(field.get());
  }
  if (lhs.isEmpty()) {
    return rhs;
  }
  if (rhs.isEmpty()) {
    return lhs;
  }
  if (lhs.isDegenerate() || rhs.isDegenerate()) {
    return lhs.toUnreduced().doUnion(rhs.toUnreduced()).reduce(field.get());
  }

  // More complex cases
  if (areOneOf<
          {Type::TypeA, Type::TypeA}, {Type::TypeB, Type::TypeB}, {Type::TypeC, Type::TypeC},
          {Type::TypeA, Type::TypeC}, {Type::TypeB, Type::TypeC}>(lhs, rhs)) {
    return Interval(rhs.ty, field.get(), std::min(lhs.a, rhs.a), std::max(lhs.b, rhs.b));
  }
  if (areOneOf<{Type::TypeA, Type::TypeB}>(lhs, rhs)) {
    auto lhsUnred = lhs.firstUnreduced();
    auto opt1 = rhs.firstUnreduced().doUnion(lhsUnred);
    auto opt2 = rhs.secondUnreduced().doUnion(lhsUnred);
    if (opt1.width() <= opt2.width()) {
      return opt1.reduce(field.get());
    }
    return opt2.reduce(field.get());
  }
  if (areOneOf<{Type::TypeF, Type::TypeF}, {Type::TypeA, Type::TypeF}>(lhs, rhs)) {
    return lhs.firstUnreduced().doUnion(rhs.firstUnreduced()).reduce(field.get());
  }
  if (areOneOf<{Type::TypeB, Type::TypeF}>(lhs, rhs)) {
    return lhs.secondUnreduced().doUnion(rhs.firstUnreduced()).reduce(field.get());
  }
  if (areOneOf<{Type::TypeC, Type::TypeF}>(lhs, rhs)) {
    return Interval::Entire(field.get());
  }
  if (areOneOf<
          {Type::TypeB, Type::TypeA}, {Type::TypeC, Type::TypeA}, {Type::TypeC, Type::TypeB},
          {Type::TypeF, Type::TypeA}, {Type::TypeF, Type::TypeB}, {Type::TypeF, Type::TypeC}>(
          lhs, rhs
      )) {
    return rhs.join(lhs);
  }
  llvm::report_fatal_error("unhandled join case");
  return Interval::Entire(field.get());
}

Interval Interval::intersect(const Interval &rhs) const {
  auto &lhs = *this;

  // Trivial cases
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return Interval::Empty(field.get());
  }
  if (lhs.isEntire()) {
    return rhs;
  }
  if (rhs.isEntire()) {
    return lhs;
  }
  if (lhs.isDegenerate() || rhs.isDegenerate()) {
    return lhs.toUnreduced().intersect(rhs.toUnreduced()).reduce(field.get());
  }

  // More complex cases
  if (areOneOf<
          {Type::TypeA, Type::TypeA}, {Type::TypeB, Type::TypeB}, {Type::TypeC, Type::TypeC},
          {Type::TypeA, Type::TypeC}, {Type::TypeB, Type::TypeC}>(lhs, rhs)) {
    auto maxA = std::max(lhs.a, rhs.a);
    auto minB = std::min(lhs.b, rhs.b);
    if (maxA <= minB) {
      return Interval(lhs.ty, field.get(), maxA, minB);
    } else {
      return Interval::Empty(field.get());
    }
  }
  if (areOneOf<{Type::TypeA, Type::TypeB}>(lhs, rhs)) {
    return Interval::Empty(field.get());
  }
  if (areOneOf<{Type::TypeF, Type::TypeF}, {Type::TypeA, Type::TypeF}>(lhs, rhs)) {
    return lhs.firstUnreduced().intersect(rhs.firstUnreduced()).reduce(field.get());
  }
  if (areOneOf<{Type::TypeB, Type::TypeF}>(lhs, rhs)) {
    return lhs.secondUnreduced().intersect(rhs.firstUnreduced()).reduce(field.get());
  }
  if (areOneOf<{Type::TypeC, Type::TypeF}>(lhs, rhs)) {
    auto rhsUnred = rhs.firstUnreduced();
    auto opt1 = lhs.firstUnreduced().intersect(rhsUnred).reduce(field.get());
    auto opt2 = lhs.secondUnreduced().intersect(rhsUnred).reduce(field.get());
    ensure(!opt1.isEntire() && !opt2.isEntire(), "impossible intersection");
    if (opt1.isEmpty()) {
      return opt2;
    }
    if (opt2.isEmpty()) {
      return opt1;
    }
    return opt1.join(opt2);
  }
  if (areOneOf<
          {Type::TypeB, Type::TypeA}, {Type::TypeC, Type::TypeA}, {Type::TypeC, Type::TypeB},
          {Type::TypeF, Type::TypeA}, {Type::TypeF, Type::TypeB}, {Type::TypeF, Type::TypeC}>(
          lhs, rhs
      )) {
    return rhs.intersect(lhs);
  }
  return Interval::Empty(field.get());
}

Interval Interval::difference(const Interval &other) const {
  Interval intersection = intersect(other);
  if (intersection.isEmpty()) {
    // There's nothing to remove, so just return this
    return *this;
  }

  const Field &f = field.get();

  // Trivial cases with a non-empty intersection
  if (isDegenerate() || other.isEntire()) {
    return Interval::Empty(f);
  }
  if (isEntire()) {
    // Since we don't support punching arbitrary holes in ranges, we only reduce
    // entire ranges if other is [0, b] or [a, prime - 1]
    if (other.a == f.zero()) {
      return UnreducedInterval(other.b + f.one(), f.maxVal()).reduce(f);
    }
    if (other.b == f.maxVal()) {
      return UnreducedInterval(f.zero(), other.a - f.one()).reduce(f);
    }

    return *this;
  }

  // Non-trivial cases
  // - Internal+internal or external+external cases
  if ((is<Type::TypeA, Type::TypeB, Type::TypeC>() &&
       intersection.is<Type::TypeA, Type::TypeB, Type::TypeC>()) ||
      areOneOf<{Type::TypeF, Type::TypeF}>(*this, intersection)) {
    // The intersection needs to be at the end of the interval, otherwise we would
    // split the interval in two, and we aren't set up to support multiple intervals
    // per value.
    if (a != intersection.a || b != intersection.b) {
      return *this;
    }
    // Otherwise, remove the intersection and reduce
    if (a == intersection.a) {
      return UnreducedInterval(intersection.b + f.one(), b).reduce(f);
    }
    // else b == intersection.b
    return UnreducedInterval(a, intersection.a - f.one()).reduce(f);
  }
  // - Mixed internal/external cases. We flip the comparison
  if (isTypeF()) {
    if (a != intersection.b || b != intersection.a) {
      return *this;
    }
    // Otherwise, remove the intersection and reduce
    if (a == intersection.b) {
      return UnreducedInterval(intersection.a + f.one(), b).reduce(f);
    }
    // else b == intersection.a
    return UnreducedInterval(a, intersection.b - f.one()).reduce(f);
  }

  // In cases we don't know how to handle, we over-approximate and return
  // the original interval.
  return *this;
}

Interval Interval::operator-() const { return (-firstUnreduced()).reduce(field.get()); }

Interval operator+(const Interval &lhs, const Interval &rhs) {
  ensure(lhs.field.get() == rhs.field.get(), "cannot add intervals in different fields");
  return (lhs.firstUnreduced() + rhs.firstUnreduced()).reduce(lhs.field.get());
}

Interval operator-(const Interval &lhs, const Interval &rhs) { return lhs + (-rhs); }

Interval operator*(const Interval &lhs, const Interval &rhs) {
  ensure(lhs.field.get() == rhs.field.get(), "cannot multiply intervals in different fields");
  const auto &field = lhs.field.get();
  auto zeroInterval = Interval::Degenerate(field, field.zero());
  if (lhs == zeroInterval || rhs == zeroInterval) {
    return zeroInterval;
  }
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return Interval::Empty(field);
  }
  if (lhs.isEntire() || rhs.isEntire()) {
    return Interval::Entire(field);
  }

  if (Interval::areOneOf<{Interval::Type::TypeB, Interval::Type::TypeB}>(lhs, rhs)) {
    return (lhs.secondUnreduced() * rhs.secondUnreduced()).reduce(field);
  }
  return (lhs.firstUnreduced() * rhs.firstUnreduced()).reduce(field);
}

Interval operator/(const Interval &lhs, const Interval &rhs) {
  const auto &field = rhs.getField();
  if (rhs.width() > field.one()) {
    return Interval::Entire(field);
  }
  if (rhs.a.isZero()) {
    llvm::report_fatal_error(
        "LLZK error in " + mlir::Twine(__PRETTY_FUNCTION__) + ": division by zero"
    );
  }
  return UnreducedInterval(lhs.a / rhs.a, lhs.b / rhs.a).reduce(field);
}

Interval operator%(const Interval &lhs, const Interval &rhs) {
  const auto &field = rhs.getField();
  return UnreducedInterval(field.zero(), rhs.b).reduce(field);
}

void Interval::print(mlir::raw_ostream &os) const {
  os << TypeName(ty);
  if (is<Type::Degenerate>()) {
    os << '(' << a << ')';
  } else if (!is<Type::Entire, Type::Empty>()) {
    os << ":[ " << a << ", " << b << " ]";
  }
}

/* ExpressionValue */

bool ExpressionValue::operator==(const ExpressionValue &rhs) const {
  if (expr == nullptr && rhs.expr == nullptr) {
    return i == rhs.i;
  }
  if (expr == nullptr || rhs.expr == nullptr) {
    return false;
  }
  return i == rhs.i && *expr == *rhs.expr;
}

ExpressionValue
intersection(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  Interval res = lhs.i.intersect(rhs.i);
  auto exprEq = solver->mkEqual(lhs.expr, rhs.expr);
  return ExpressionValue(exprEq, res);
}

ExpressionValue
add(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = lhs.i + rhs.i;
  res.expr = solver->mkBVAdd(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
sub(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = lhs.i - rhs.i;
  res.expr = solver->mkBVSub(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
mul(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = lhs.i * rhs.i;
  res.expr = solver->mkBVMul(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
div(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = lhs.i / rhs.i;
  res.expr = solver->mkBVUDiv(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
mod(llvm::SMTSolverRef solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = lhs.i % rhs.i;
  res.expr = solver->mkBVURem(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
cmp(llvm::SMTSolverRef solver, CmpOp op, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = Interval::Boolean(lhs.getField());
  switch (op.getPredicate()) {
  case FeltCmpPredicate::EQ:
    res.expr = solver->mkEqual(lhs.expr, rhs.expr);
    res.i = lhs.i.intersect(rhs.i);
    break;
  case FeltCmpPredicate::NE:
    res.expr = solver->mkNot(solver->mkEqual(lhs.expr, rhs.expr));
    break;
  case FeltCmpPredicate::LT:
    res.expr = solver->mkBVUlt(lhs.expr, rhs.expr);
    break;
  case FeltCmpPredicate::LE:
    res.expr = solver->mkBVUle(lhs.expr, rhs.expr);
    break;
  case FeltCmpPredicate::GT:
    res.expr = solver->mkBVUgt(lhs.expr, rhs.expr);
    break;
  case FeltCmpPredicate::GE:
    res.expr = solver->mkBVUge(lhs.expr, rhs.expr);
    break;
  }
  return res;
}

ExpressionValue fallbackBinaryOp(
    llvm::SMTSolverRef solver, mlir::Operation *op, const ExpressionValue &lhs,
    const ExpressionValue &rhs
) {
  ExpressionValue res;
  res.i = Interval::Entire(lhs.getField());
  res.expr = TypeSwitch<Operation *, llvm::SMTExprRef>(op)
                 .Case<AndFeltOp>([&](AndFeltOp _) { return solver->mkBVAnd(lhs.expr, rhs.expr); })
                 .Case<OrFeltOp>([&](OrFeltOp _) { return solver->mkBVOr(lhs.expr, rhs.expr); })
                 .Case<XorFeltOp>([&](XorFeltOp _) { return solver->mkBVXor(lhs.expr, rhs.expr); })
                 .Case<ShlFeltOp>([&](ShlFeltOp _) { return solver->mkBVShl(lhs.expr, rhs.expr); })
                 .Case<ShrFeltOp>([&](ShrFeltOp _) {
    return solver->mkBVLshr(lhs.expr, rhs.expr);
  }).Default([&](Operation *unsupported) {
    llvm::report_fatal_error(
        "no fallback provided for " + mlir::Twine(unsupported->getName().getStringRef())
    );
    return nullptr;
  });

  return res;
}

ExpressionValue neg(llvm::SMTSolverRef solver, const ExpressionValue &val) {
  ExpressionValue res;
  res.i = -val.i;
  res.expr = solver->mkBVNeg(val.expr);
  return res;
}

ExpressionValue notOp(llvm::SMTSolverRef solver, const ExpressionValue &val) {
  ExpressionValue res;
  const auto &f = val.getField();
  if (val.i.isDegenerate()) {
    if (val.i == Interval::Degenerate(f, f.zero())) {
      res.i = Interval::Degenerate(f, f.one());
    } else {
      res.i = Interval::Degenerate(f, f.zero());
    }
  }
  res.i = Interval::Boolean(f);
  res.expr = solver->mkBVNot(val.expr);
  return res;
}

ExpressionValue
fallbackUnaryOp(llvm::SMTSolverRef solver, mlir::Operation *op, const ExpressionValue &val) {
  const Field &field = val.getField();
  ExpressionValue res;
  res.i = Interval::Entire(field);
  res.expr = TypeSwitch<Operation *, llvm::SMTExprRef>(op)
                 .Case<InvFeltOp>([&](InvFeltOp _) {
    // The definition of an inverse X^-1 is Y s.t. XY % prime = 1.
    // To create this expression, we create a new symbol for Y and add the
    // XY % prime = 1 constraint to the solver.
    std::string symName;
    llvm::raw_string_ostream ss(symName);
    ss << *op;

    llvm::SMTExprRef invSym = field.createSymbol(solver, symName.c_str());
    llvm::SMTExprRef one = solver->mkBitvector(field.one(), field.bitWidth());
    llvm::SMTExprRef prime = solver->mkBitvector(field.prime(), field.bitWidth());
    llvm::SMTExprRef mult = solver->mkBVMul(val.getExpr(), invSym);
    llvm::SMTExprRef mod = solver->mkBVURem(mult, prime);
    llvm::SMTExprRef constraint = solver->mkEqual(mod, one);
    solver->addConstraint(constraint);
    return invSym;
  }).Default([&](Operation *unsupported) {
    llvm::report_fatal_error(
        "no fallback provided for " + mlir::Twine(op->getName().getStringRef())
    );
    return nullptr;
  });

  return res;
}

void ExpressionValue::print(mlir::raw_ostream &os) const {
  if (expr) {
    expr->print(os);
  } else {
    os << "<null expression>";
  }

  os << " ( interval: " << i << " )";
}

/* IntervalAnalysisLattice */

mlir::ChangeResult IntervalAnalysisLattice::join(const AbstractDenseLattice &other) {
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
    if (!constraints.contains(v)) {
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

void IntervalAnalysisLattice::print(mlir::raw_ostream &os) const {
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

mlir::FailureOr<IntervalAnalysisLattice::LatticeValue>
IntervalAnalysisLattice::getValue(mlir::Value v) const {
  auto it = valMap.find(v);
  if (it == valMap.end()) {
    return mlir::failure();
  }
  return it->second;
}

mlir::ChangeResult IntervalAnalysisLattice::setValue(mlir::Value v, ExpressionValue e) {
  LatticeValue val(e);
  if (valMap[v] == val) {
    return mlir::ChangeResult::NoChange;
  }
  valMap[v] = val;
  intervals[e.getExpr()] = e.getInterval();
  return mlir::ChangeResult::Change;
}

mlir::ChangeResult IntervalAnalysisLattice::addSolverConstraint(ExpressionValue e) {
  if (!constraints.contains(e)) {
    constraints.insert(e);
    return mlir::ChangeResult::Change;
  }
  return mlir::ChangeResult::NoChange;
}

mlir::FailureOr<Interval> IntervalAnalysisLattice::findInterval(llvm::SMTExprRef expr) const {
  auto it = intervals.find(expr);
  if (it != intervals.end()) {
    return it->second;
  }
  return mlir::failure();
}

/* IntervalDataFlowAnalysis */

/// @brief The interval analysis is intraprocedural only for now, so this control
/// flow transfer function passes no data to the callee and sets the post-call
/// state to that of the pre-call state (i.e., calls are ignored).
void IntervalDataFlowAnalysis::visitCallControlFlowTransfer(
    mlir::CallOpInterface call, dataflow::CallControlFlowAction action,
    const IntervalAnalysisLattice &before, IntervalAnalysisLattice *after
) {
  /// `action == CallControlFlowAction::Enter` indicates that:
  ///   - `before` is the state before the call operation;
  ///   - `after` is the state at the beginning of the callee entry block;
  if (action == dataflow::CallControlFlowAction::EnterCallee) {
    // We skip updating the incoming lattice for function calls,
    // as values are relative to the containing function/struct, so we don't need to pollute
    // the callee with the callers values.
    setToEntryState(after);
  }
  /// `action == CallControlFlowAction::Exit` indicates that:
  ///   - `before` is the state at the end of a callee exit block;
  ///   - `after` is the state after the call operation.
  else if (action == dataflow::CallControlFlowAction::ExitCallee) {
    // Get the argument values of the lattice by getting the state as it would
    // have been for the callsite.
    dataflow::AbstractDenseLattice *beforeCall = nullptr;
    if (auto *prev = call->getPrevNode()) {
      beforeCall = getLattice(prev);
    } else {
      beforeCall = getLattice(call->getBlock());
    }
    ensure(beforeCall, "could not get prior lattice");

    // The lattice at the return is the lattice before the call
    propagateIfChanged(after, after->join(*beforeCall));
  }
  /// `action == CallControlFlowAction::External` indicates that:
  ///   - `before` is the state before the call operation.
  ///   - `after` is the state after the call operation, since there is no callee
  ///      body to enter into.
  else if (action == mlir::dataflow::CallControlFlowAction::ExternalCallee) {
    // For external calls, we propagate what information we already have from
    // before the call to after the call, since the external call won't invalidate
    // any of that information. It also, conservatively, makes no assumptions about
    // external calls and their computation, so CDG edges will not be computed over
    // input arguments to external functions.
    join(after, before);
  }
}

void IntervalDataFlowAnalysis::visitOperation(
    mlir::Operation *op, const Lattice &before, Lattice *after
) {
  mlir::ChangeResult changed = after->join(before);

  llvm::SmallVector<LatticeValue> operandVals;

  auto constrainRefLattice = dataflowSolver.lookupState<ConstrainRefLattice>(op);
  ensure(constrainRefLattice, "failed to get lattice");

  for (auto &operand : op->getOpOperands()) {
    auto val = operand.get();
    // First, lookup the operand value in the before state.
    auto priorState = before.getValue(val);
    if (mlir::succeeded(priorState)) {
      operandVals.push_back(*priorState);
      continue;
    }
    // Else, look up the stored value by constrain ref.
    // We only care about scalar type values, so we ignore composite types, which
    // are currently limited to structs and arrays.
    if (mlir::isa<StructType, ArrayType>(val.getType())) {
      operandVals.push_back(LatticeValue());
      continue;
    }

    ConstrainRefLatticeValue refSet = constrainRefLattice->getOrDefault(val);
    ensure(refSet.isScalar(), "should have ruled out array values already");

    if (refSet.getScalarValue().empty()) {
      // If we can't compute the reference, then there must be some unsupported
      // op the reference analysis cannot handle. We emit a warning and return
      // early, since there's no meaningful computation we can do for this op.
      std::string warning;
      debug::Appender(warning
      ) << "state of "
        << val << " is empty; defining operation is unsupported by constrain ref analysis";
      op->emitWarning(warning);
      return;
    } else if (!refSet.isSingleValue()) {
      std::string warning;
      debug::Appender(warning) << "operand " << val << " is not a single value " << refSet
                               << ", overapproximating";
      op->emitWarning(warning);
      operandVals.push_back(LatticeValue());
    } else {
      auto ref = refSet.getSingleValue();
      auto exprVal = ExpressionValue(field.get(), getOrCreateSymbol(ref));
      changed |= after->setValue(val, exprVal);
      operandVals.emplace_back(exprVal);
    }
  }

  // Now, the way we update is dependent on the type of the operation.
  if (!isConsideredOp(op)) {
    op->emitWarning("unconsidered operation type, analysis may be incomplete");
  }

  if (isConstOp(op)) {
    auto constVal = getConst(op);
    auto expr = createConstBitvectorExpr(constVal);
    auto latticeVal = ExpressionValue(field.get(), expr, constVal);
    changed |= after->setValue(op->getResult(0), latticeVal);
  } else if (isArithmeticOp(op)) {
    ensure(operandVals.size() <= 2, "arithmetic op with the wrong number of operands");
    ExpressionValue result;
    if (operandVals.size() == 2) {
      result = performBinaryArithmetic(op, operandVals[0], operandVals[1]);
    } else {
      result = performUnaryArithmetic(op, operandVals[0]);
    }

    changed |= after->setValue(op->getResult(0), result);
  } else if (mlir::isa<EmitEqualityOp>(op)) {
    ensure(operandVals.size() == 2, "constraint op with the wrong number of operands");
    auto lhsVal = op->getOperand(0);
    auto rhsVal = op->getOperand(1);
    auto lhsExpr = operandVals[0].getScalarValue();
    auto rhsExpr = operandVals[1].getScalarValue();

    auto constraint = intersection(smtSolver, lhsExpr, rhsExpr);
    // Update the LHS and RHS to the same value, but restricted intervals
    // based on the constraints
    changed |= applyInterval(after, lhsVal, constraint.getInterval());
    changed |= applyInterval(after, rhsVal, constraint.getInterval());
    changed |= after->addSolverConstraint(constraint);
  } else if (isAssertOp(op)) {
    ensure(operandVals.size() == 1, "assert op with the wrong number of operands");
    // assert enforces that the operand is true. So we apply an interval of [1, 1]
    // to the operand.
    changed |= applyInterval(
        after, op->getOperand(0), Interval::Degenerate(field.get(), field.get().one())
    );
    // Also add the solver constraint that the expression must be true.
    auto assertExpr = operandVals[0].getScalarValue();
    changed |= after->addSolverConstraint(assertExpr);
  } else if (!isReadOp(op)          /* We do not need to explicitly handle read ops
                      since they are resolved at the operand value step where constrain refs are
                      queries */
             && !isReturnOp(op)     /* We do not currently handle return ops as the analysis
                 is currently limited to constrain functions, which return no value. */
             && !isDefinitionOp(op) /* The analysis ignores field, struct, function definitions. */
             &&
             !mlir::isa<CreateStructOp>(op) /* We do not need to analyze the creation of structs. */
  ) {
    op->emitWarning("unhandled operation, analysis may be incomplete");
  }

  propagateIfChanged(after, changed);
}

llvm::SMTExprRef IntervalDataFlowAnalysis::getOrCreateSymbol(const ConstrainRef &r) {
  auto it = refSymbols.find(r);
  if (it != refSymbols.end()) {
    return it->second;
  }
  auto sym = createFeltSymbol(r);
  refSymbols[r] = sym;
  return sym;
}

llvm::SMTExprRef IntervalDataFlowAnalysis::createFeltSymbol(const ConstrainRef &r) const {
  std::string symbolName;
  llvm::raw_string_ostream ss(symbolName);
  r.print(ss);

  return createFeltSymbol(symbolName.c_str());
}

llvm::SMTExprRef IntervalDataFlowAnalysis::createFeltSymbol(mlir::Value val) const {
  std::string symbolName;
  llvm::raw_string_ostream ss(symbolName);
  val.print(ss);

  return createFeltSymbol(symbolName.c_str());
}

llvm::SMTExprRef IntervalDataFlowAnalysis::createFeltSymbol(const char *name) const {
  return field.get().createSymbol(smtSolver, name);
}

llvm::APSInt IntervalDataFlowAnalysis::getConst(mlir::Operation *op) const {
  ensure(isConstOp(op), "op is not a const op");

  llvm::APInt fieldConst =
      TypeSwitch<Operation *, llvm::APInt>(op)
          .Case<FeltConstantOp>([&](FeltConstantOp feltConst) {
    return feltConst.getValueAttr().getValue().zext(field.get().bitWidth());
  })
          .Case<mlir::arith::ConstantIndexOp>([&](mlir::arith::ConstantIndexOp indexConst) {
    return llvm::APInt(field.get().bitWidth(), indexConst.value());
  }).Default([&](Operation *illegalOp) {
    std::string err;
    debug::Appender(err) << "unhandled getConst case: " << *illegalOp;
    llvm::report_fatal_error(Twine(err));
    return llvm::APInt();
  });
  return llvm::APSInt(fieldConst);
}

ExpressionValue IntervalDataFlowAnalysis::performBinaryArithmetic(
    mlir::Operation *op, const LatticeValue &a, const LatticeValue &b
) {
  ensure(isArithmeticOp(op), "is not arithmetic op");

  auto lhs = a.getScalarValue(), rhs = b.getScalarValue();
  ensure(lhs.getExpr(), "cannot perform arithmetic over null lhs smt expr");
  ensure(rhs.getExpr(), "cannot perform arithmetic over null rhs smt expr");

  auto res = TypeSwitch<Operation *, ExpressionValue>(op)
                 .Case<AddFeltOp>([&](AddFeltOp _) { return add(smtSolver, lhs, rhs); })
                 .Case<SubFeltOp>([&](SubFeltOp _) { return sub(smtSolver, lhs, rhs); })
                 .Case<MulFeltOp>([&](MulFeltOp _) { return mul(smtSolver, lhs, rhs); })
                 .Case<DivFeltOp>([&](DivFeltOp _) { return div(smtSolver, lhs, rhs); })
                 .Case<ModFeltOp>([&](ModFeltOp _) { return mod(smtSolver, lhs, rhs); })
                 .Case<CmpOp>([&](CmpOp cmpOp) {
    return cmp(smtSolver, cmpOp, lhs, rhs);
  }).Default([&](Operation *unsupported) {
    unsupported->emitWarning(
        "unsupported binary arithmetic operation, defaulting to over-approximated intervals"
    );
    return fallbackBinaryOp(smtSolver, unsupported, lhs, rhs);
  });

  ensure(res.getExpr(), "arithmetic produced null smt expr");
  return res;
}

ExpressionValue
IntervalDataFlowAnalysis::performUnaryArithmetic(mlir::Operation *op, const LatticeValue &a) {
  ensure(isArithmeticOp(op), "is not arithmetic op");

  auto val = a.getScalarValue();
  ensure(val.getExpr(), "cannot perform arithmetic over null smt expr");

  auto res = TypeSwitch<Operation *, ExpressionValue>(op)
                 .Case<NegFeltOp>([&](NegFeltOp _) { return neg(smtSolver, val); })
                 .Case<NotFeltOp>([&](NotFeltOp _) {
    return notOp(smtSolver, val);
  }).Default([&](Operation *unsupported) {
    unsupported->emitWarning(
        "unsupported unary arithmetic operation, defaulting to over-approximated interval"
    );
    return fallbackUnaryOp(smtSolver, unsupported, val);
  });

  ensure(res.getExpr(), "arithmetic produced null smt expr");
  return res;
}

ChangeResult
IntervalDataFlowAnalysis::applyInterval(Lattice *after, Value val, Interval newInterval) {
  auto latValRes = after->getValue(val);
  if (failed(latValRes)) {
    // visitOperation didn't add val to the lattice, so there's nothing to do
    return ChangeResult::NoChange;
  }
  ChangeResult res = after->setValue(val, latValRes->getScalarValue().withInterval(newInterval));
  // Now we descend into val's operands, if it has any.
  Operation *definingOp = val.getDefiningOp();
  if (!definingOp) {
    return res;
  }

  const Field &f = field.get();

  // This is a rules-based operation. If we have a rule for a given operation,
  // then we can make some kind of update, otherwise we leave the intervals
  // as is.
  // - First we'll define all the rules so the type switch can be less messy

  // cmp.<pred> restricts each side of the comparison if the result is known.
  auto cmpCase = [&](CmpOp cmpOp) {
    // Cmp output range is [0, 1], so in order to do something, we must have newInterval
    // either "true" (1) or "false" (0)
    Interval maxInterval = Interval::Boolean(f);
    ensure(
        newInterval.intersect(maxInterval).isNotEmpty(),
        "new interval for CmpOp outside of allowed boolean range or is empty"
    );
    if (!newInterval.isDegenerate()) {
      // The comparison result is unknown, so we can't update the operand ranges
      return ChangeResult::NoChange;
    }

    bool cmpTrue = newInterval.rhs() == f.one();

    Value lhs = cmpOp->getOperand(0), rhs = cmpOp->getOperand(1);
    auto lhsLatValRes = after->getValue(lhs), rhsLatValRes = after->getValue(rhs);
    if (failed(lhsLatValRes) || failed(rhsLatValRes)) {
      return ChangeResult::NoChange;
    }
    ExpressionValue lhsExpr = lhsLatValRes->getScalarValue(),
                    rhsExpr = rhsLatValRes->getScalarValue();

    Interval newLhsInterval, newRhsInterval;
    const Interval &lhsInterval = lhsExpr.getInterval();
    const Interval &rhsInterval = rhsExpr.getInterval();

    FeltCmpPredicate pred = cmpOp.getPredicate();
    // predicate cases
    auto eqCase = [&]() {
      return (pred == FeltCmpPredicate::EQ && cmpTrue) ||
             (pred == FeltCmpPredicate::NE && !cmpTrue);
    };
    auto neCase = [&]() {
      return (pred == FeltCmpPredicate::NE && cmpTrue) ||
             (pred == FeltCmpPredicate::EQ && !cmpTrue);
    };
    auto ltCase = [&]() {
      return (pred == FeltCmpPredicate::LT && cmpTrue) ||
             (pred == FeltCmpPredicate::GE && !cmpTrue);
    };
    auto leCase = [&]() {
      return (pred == FeltCmpPredicate::LE && cmpTrue) ||
             (pred == FeltCmpPredicate::GT && !cmpTrue);
    };
    auto gtCase = [&]() {
      return (pred == FeltCmpPredicate::GT && cmpTrue) ||
             (pred == FeltCmpPredicate::LE && !cmpTrue);
    };
    auto geCase = [&]() {
      return (pred == FeltCmpPredicate::GE && cmpTrue) ||
             (pred == FeltCmpPredicate::LT && !cmpTrue);
    };

    // new intervals based on case
    if (eqCase()) {
      newLhsInterval = newRhsInterval = lhsInterval.intersect(rhsInterval);
    } else if (neCase()) {
      if (lhsInterval.isDegenerate() && rhsInterval.isDegenerate() && lhsInterval == rhsInterval) {
        // In this case, we know lhs and rhs cannot satisfy this assertion, so they have
        // an empty value range.
        newLhsInterval = newRhsInterval = Interval::Empty(f);
      } else {
        // Leave unchanged
        newLhsInterval = lhsInterval;
        newRhsInterval = rhsInterval;
      }
    } else if (ltCase()) {
      newLhsInterval = lhsInterval.toUnreduced().lt(rhsInterval.toUnreduced()).reduce(f);
      newRhsInterval = rhsInterval.toUnreduced().ge(lhsInterval.toUnreduced()).reduce(f);
    } else if (leCase()) {
      newLhsInterval = lhsInterval.toUnreduced().le(rhsInterval.toUnreduced()).reduce(f);
      newRhsInterval = rhsInterval.toUnreduced().gt(lhsInterval.toUnreduced()).reduce(f);
    } else if (gtCase()) {
      newLhsInterval = lhsInterval.toUnreduced().gt(rhsInterval.toUnreduced()).reduce(f);
      newRhsInterval = rhsInterval.toUnreduced().le(lhsInterval.toUnreduced()).reduce(f);
    } else if (geCase()) {
      newLhsInterval = lhsInterval.toUnreduced().ge(rhsInterval.toUnreduced()).reduce(f);
      newRhsInterval = rhsInterval.toUnreduced().lt(lhsInterval.toUnreduced()).reduce(f);
    } else {
      cmpOp->emitWarning("unhandled cmp predicate");
      return ChangeResult::NoChange;
    }

    // Now we recurse to each operand
    return applyInterval(after, lhs, newLhsInterval) | applyInterval(after, rhs, newRhsInterval);
  };

  // If the result of a multiplication is non-zero, then both operands must be
  // non-zero.
  auto mulCase = [&](MulFeltOp mulOp) {
    auto zeroInt = Interval::Degenerate(f, f.zero());
    if (newInterval.intersect(zeroInt).isNotEmpty()) {
      // The multiplication may be zero, so we can't reduce the operands to be non-zero
      return ChangeResult::NoChange;
    }

    Value lhs = mulOp->getOperand(0), rhs = mulOp->getOperand(1);
    auto lhsLatValRes = after->getValue(lhs), rhsLatValRes = after->getValue(rhs);
    if (failed(lhsLatValRes) || failed(rhsLatValRes)) {
      return ChangeResult::NoChange;
    }
    ExpressionValue lhsExpr = lhsLatValRes->getScalarValue(),
                    rhsExpr = rhsLatValRes->getScalarValue();
    Interval newLhsInterval = lhsExpr.getInterval().difference(zeroInt);
    Interval newRhsInterval = rhsExpr.getInterval().difference(zeroInt);
    return applyInterval(after, lhs, newLhsInterval) | applyInterval(after, rhs, newRhsInterval);
  };

  // - Apply the rules given the op.
  res |= TypeSwitch<Operation *, ChangeResult>(definingOp)
             .Case<CmpOp>([&](CmpOp op) { return cmpCase(op); })
             .Case<MulFeltOp>([&](MulFeltOp op) {
    return mulCase(op);
  }).Default([&](Operation *_) { return ChangeResult::NoChange; });

  return res;
}

/* StructIntervals */

mlir::LogicalResult StructIntervals::computeIntervals(
    mlir::DataFlowSolver &solver, mlir::AnalysisManager &am, IntervalAnalysisContext &ctx
) {
  // Get the lattice at the end of the constrain function.
  ReturnOp constrainEnd;
  structDef.getConstrainFuncOp().walk([&constrainEnd](ReturnOp r) mutable { constrainEnd = r; });

  auto constrainLattice = solver.lookupState<IntervalAnalysisLattice>(constrainEnd);

  constrainSolverConstraints = constrainLattice->getConstraints();

  for (const auto &ref : ConstrainRef::getAllConstrainRefs(structDef)) {
    // We only want to compute intervals for field elements and not composite types
    if (!ref.isScalar()) {
      continue;
    }
    auto symbol = ctx.getSymbol(ref);
    auto constrainInterval = constrainLattice->findInterval(symbol);
    if (mlir::succeeded(constrainInterval)) {
      constrainFieldRanges[ref] = *constrainInterval;
    } else {
      constrainFieldRanges[ref] = Interval::Entire(ctx.field);
    }
  }

  return mlir::success();
}

void StructIntervals::print(mlir::raw_ostream &os, bool withConstraints) const {
  os << "StructIntervals { ";
  if (constrainFieldRanges.empty()) {
    os << "}\n";
    return;
  }

  for (auto &[ref, interval] : constrainFieldRanges) {
    os << "\n    " << ref << " in " << interval;
  }

  if (withConstraints) {
    os << "\n\n    Solver Constraints { ";
    if (constrainSolverConstraints.empty()) {
      os << "}\n";
    } else {
      for (const auto &e : constrainSolverConstraints) {
        os << "\n        ";
        e.getExpr()->print(os);
      }
      os << "\n    }";
    }
  }

  os << "\n}\n";
}

} // namespace llzk
