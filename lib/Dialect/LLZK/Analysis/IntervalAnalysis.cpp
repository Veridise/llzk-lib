#include "llzk/Dialect/LLZK/Analysis/IntervalAnalysis.h"

namespace llzk {

/* UnreducedInterval */

Interval UnreducedInterval::reduce(const Field &field) const {
  if (a > b) {
    return Interval::Empty();
  }
  if (width() >= field.prime()) {
    return Interval::Entire();
  }

  auto lhs = field.reduce(a), rhs = field.reduce(b);

  if ((rhs - lhs).isZero()) {
    return Interval::Degenerate(lhs);
  }

  const auto &half = field.half();
  if (lhs.ule(rhs)) {
    if (lhs.ult(half) && rhs.ult(half)) {
      return Interval::TypeA(lhs, rhs);
    } else if (lhs.ult(half)) {
      return Interval::TypeC(lhs, rhs);
    } else {
      return Interval::TypeB(lhs, rhs);
    }
  } else {
    if (lhs.uge(half) && rhs.ult(half)) {
      return Interval::TypeF(lhs, rhs);
    } else {
      return Interval::Entire();
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
  auto newI = le(rhs);
  auto one = llvm::APInt(newI.a.getBitWidth(), 1);
  return UnreducedInterval(newI.a, newI.b - llvm::APSInt(one));
}

UnreducedInterval UnreducedInterval::le(const UnreducedInterval &rhs) const {
  auto &lhs = *this;
  return UnreducedInterval(lhs.a, std::min(lhs.b, rhs.b));
}

UnreducedInterval UnreducedInterval::gt(const UnreducedInterval &rhs) const {
  auto newI = ge(rhs);
  auto one = llvm::APInt(newI.a.getBitWidth(), 1);
  return UnreducedInterval(newI.a + llvm::APSInt(one), newI.b);
}

UnreducedInterval UnreducedInterval::ge(const UnreducedInterval &rhs) const {
  auto &lhs = *this;
  return UnreducedInterval(std::max(lhs.a, rhs.a), lhs.b);
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
    return Interval::Empty();
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
    return Interval(rhs.ty, std::min(lhs.a, rhs.a), std::max(lhs.b, rhs.b));
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
    return Interval::Entire();
  }
  if (areOneOf<
          {Type::TypeB, Type::TypeA}, {Type::TypeC, Type::TypeA}, {Type::TypeC, Type::TypeB},
          {Type::TypeF, Type::TypeA}, {Type::TypeF, Type::TypeB}, {Type::TypeF, Type::TypeC}>(
          lhs, rhs
      )) {
    return rhs.join(lhs);
  }
  llvm::report_fatal_error("unhandled case");
  return Interval::Entire();
}

Interval Interval::intersect(const Interval &rhs) const {
  auto &lhs = *this;

  // Trivial cases
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return Interval::Empty();
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
      return Interval(lhs.ty, a, b);
    } else {
      return Interval::Empty();
    }
  }
  if (areOneOf<{Type::TypeA, Type::TypeB}>(lhs, rhs)) {
    return Interval::Empty();
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
  return Interval::Empty();
}

Interval Interval::operator-() const { return (-firstUnreduced()).reduce(field.get()); }

Interval operator+(const Interval &lhs, const Interval &rhs) {
  ensure(lhs.field.get() == rhs.field.get(), "cannot add intervals in different fields");
  return (lhs.firstUnreduced() + rhs.firstUnreduced()).reduce(lhs.field.get());
}

Interval operator-(const Interval &lhs, const Interval &rhs) { return lhs + (-rhs); }

Interval operator*(const Interval &lhs, const Interval &rhs) {
  ensure(lhs.field.get() == rhs.field.get(), "cannot multiply intervals in different fields");
  auto zeroInterval = Interval::Degenerate(lhs.field.get().zero());
  if (lhs == zeroInterval || rhs == zeroInterval) {
    return zeroInterval;
  }
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return Interval::Empty();
  }
  if (lhs.isEntire() || rhs.isEntire()) {
    return Interval::Entire();
  }

  if (Interval::areOneOf<{Interval::Type::TypeB, Interval::Type::TypeB}>(lhs, rhs)) {
    return (lhs.secondUnreduced() * rhs.secondUnreduced()).reduce(lhs.field.get());
  }
  return (lhs.firstUnreduced() * rhs.firstUnreduced()).reduce(lhs.field.get());
}

Interval operator/(const Interval &lhs, const Interval &rhs) {
  if (rhs.width() > rhs.field.get().one()) {
    return Interval::Entire();
  }
  if (rhs.a.isZero()) {
    return lhs;
  }
  llvm::errs() << lhs.a.isSigned() << " " << lhs.b.isSigned() << " " << rhs.a.isSigned() << "\n";
  llvm::errs() << lhs << " " << rhs << '\n';
  return UnreducedInterval(lhs.a / rhs.a, lhs.b / rhs.a).reduce(rhs.field.get());
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
      computeFieldRanges[ref] = Interval::Entire();
    }
    auto constrainInterval = constrainLattice->findInterval(symbol);
    if (mlir::succeeded(constrainInterval)) {
      constrainFieldRanges[ref] = *constrainInterval;
    } else {
      constrainFieldRanges[ref] = Interval::Entire();
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
