#include "llzk/Dialect/LLZK/Analysis/IntervalAnalysis.h"

namespace llzk {

/* UnreducedInterval */

Interval UnreducedInterval::reduce(const Field &field) const {
  llvm::errs() << "reduce: " << a.getBitWidth() << ", " << b.getBitWidth() << "\n";

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

bool UnreducedInterval::operator<(const UnreducedInterval &rhs) const {
  auto &lhs = *this;
  return lhs.a < rhs.a;
}

bool UnreducedInterval::operator>(const UnreducedInterval &rhs) const {
  auto &lhs = *this;
  return lhs.a > rhs.a;
}

bool UnreducedInterval::operator==(const UnreducedInterval &rhs) const {
  auto &lhs = *this;
  return lhs.a == rhs.a && lhs.b == rhs.b;
}

/* Interval */

Interval Interval::join(const Interval &rhs) const {
  auto &lhs = *this;

  // Trivial cases
  if (lhs.isEntire() || rhs.isEntire()) {
    return Interval::Empty(field.get());
  }
  if (lhs.isEmpty()) {
    return rhs;
  }
  if (rhs.isEmpty()) {
    return rhs;
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
  llvm::report_fatal_error("unhandled case");
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
      return Interval(lhs.ty, field.get(), a, b);
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
    return lhs;
  }
  return UnreducedInterval(lhs.a / rhs.a, lhs.b / rhs.a).reduce(field);
}

Interval operator%(const Interval &lhs, const Interval &rhs) {
  const auto &field = rhs.getField();
  return UnreducedInterval(field.zero(), rhs.b).reduce(field);
}

/* StructIntervals */

mlir::LogicalResult StructIntervals::computeIntervals(
    mlir::DataFlowSolver &solver, mlir::AnalysisManager &am, IntervalAnalysisContext &ctx
) {
  ReturnOp computeEnd, constrainEnd;
  structDef.getComputeFuncOp().walk([&computeEnd](ReturnOp r) mutable { computeEnd = r; });
  structDef.getConstrainFuncOp().walk([&constrainEnd](ReturnOp r) mutable { constrainEnd = r; });

  auto computeLattice = solver.lookupState<IntervalAnalysisLattice>(computeEnd);
  auto constrainLattice = solver.lookupState<IntervalAnalysisLattice>(constrainEnd);

  smtSolver = ctx.smtSolver;

  computeSolverConstraints = computeLattice->getConstraints();
  constrainSolverConstraints = constrainLattice->getConstraints();

  for (const auto &ref : ConstrainRef::getAllConstrainRefs(structDef)) {
    if (!ref.isScalar()) {
      continue;
    }
    auto symbol = ctx.getSymbol(ref);
    auto computeInterval = computeLattice->findInterval(symbol);
    if (mlir::succeeded(computeInterval)) {
      computeFieldRanges[ref] = *computeInterval;
    } else {
      computeFieldRanges[ref] = Interval::Entire(ctx.field);
    }
    auto constrainInterval = constrainLattice->findInterval(symbol);
    if (mlir::succeeded(constrainInterval)) {
      constrainFieldRanges[ref] = *constrainInterval;
    } else {
      constrainFieldRanges[ref] = Interval::Entire(ctx.field);
    }
  }

  return mlir::success();
}

void StructIntervals::print(mlir::raw_ostream &os) const {
  os << "StructIntervals { ";
  if (computeFieldRanges.empty() && constrainFieldRanges.empty()) {
    os << "}\n";
    return;
  }

  os << "\n    compute { ";
  for (auto &[ref, interval] : computeFieldRanges) {
    os << "\n        " << ref << " in " << interval;
  }
  os << "\n\n        Solver Constraints { ";
  if (computeSolverConstraints.empty()) {
    os << "}\n";
  } else {
    for (const auto &e : computeSolverConstraints) {
      os << "\n            " << e;
    }
    os << "\n        }\n";
  }

  os << "    }\n";

  os << "\n    constrain { ";
  for (auto &[ref, interval] : constrainFieldRanges) {
    os << "\n        " << ref << " in " << interval;
  }
  os << "\n\n        Solver Constraints { ";
  if (constrainSolverConstraints.empty()) {
    os << "}\n";
  } else {
    for (const auto &e : constrainSolverConstraints) {
      os << "\n            ";
      e.getExpr()->print(os);
    }
    os << "\n        }\n";
  }

  os << "    }\n";

  os << "}\n";
}

} // namespace llzk
