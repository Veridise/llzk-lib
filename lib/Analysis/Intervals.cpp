//===-- Intervals.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/Intervals.h"
#include "llzk/Util/APIntHelper.h"
#include "llzk/Util/ErrorHelper.h"

using namespace mlir;

namespace llzk {

/* UnreducedInterval */

Interval UnreducedInterval::reduce(const Field &field) const {
  if (safeGt(a, b)) {
    return Interval::Empty(field);
  }
  if (safeGe(width(), field.prime())) {
    return Interval::Entire(field);
  }
  auto lhs = field.reduce(a), rhs = field.reduce(b);
  // lhs and rhs are now guaranteed to have the same bitwidth, so we can use
  // built-in functions.
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
  return UnreducedInterval(safeMax(lhs.a, rhs.a), safeMin(lhs.b, rhs.b));
}

UnreducedInterval UnreducedInterval::doUnion(const UnreducedInterval &rhs) const {
  auto &lhs = *this;
  return UnreducedInterval(safeMin(lhs.a, rhs.a), safeMax(lhs.b, rhs.b));
}

UnreducedInterval UnreducedInterval::computeLTPart(const UnreducedInterval &rhs) const {
  if (isEmpty() || rhs.isEmpty()) {
    return *this;
  }
  auto one = llvm::APSInt(llvm::APInt(a.getBitWidth(), 1));
  auto bound = expandingSub(rhs.b, one);
  return UnreducedInterval(a, safeMin(b, bound));
}

UnreducedInterval UnreducedInterval::computeLEPart(const UnreducedInterval &rhs) const {
  if (isEmpty() || rhs.isEmpty()) {
    return *this;
  }
  return UnreducedInterval(a, safeMin(b, rhs.b));
}

UnreducedInterval UnreducedInterval::computeGTPart(const UnreducedInterval &rhs) const {
  if (isEmpty() || rhs.isEmpty()) {
    return *this;
  }
  auto one = llvm::APSInt(llvm::APInt(a.getBitWidth(), 1));
  auto bound = expandingAdd(rhs.a, one);
  return UnreducedInterval(safeMax(a, bound), b);
}

UnreducedInterval UnreducedInterval::computeGEPart(const UnreducedInterval &rhs) const {
  if (isEmpty() || rhs.isEmpty()) {
    return *this;
  }
  return UnreducedInterval(safeMax(a, rhs.a), b);
}

UnreducedInterval UnreducedInterval::operator-() const { return UnreducedInterval(-b, -a); }

UnreducedInterval operator+(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
  llvm::APSInt low = expandingAdd(lhs.a, rhs.a), high = expandingAdd(lhs.b, rhs.b);
  return UnreducedInterval(low, high);
}

UnreducedInterval operator-(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
  return lhs + (-rhs);
}

UnreducedInterval operator*(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
  auto v1 = expandingMul(lhs.a, rhs.a);
  auto v2 = expandingMul(lhs.a, rhs.b);
  auto v3 = expandingMul(lhs.b, rhs.a);
  auto v4 = expandingMul(lhs.b, rhs.b);

  auto minVal = safeMin({v1, v2, v3, v4});
  auto maxVal = safeMax({v1, v2, v3, v4});

  return UnreducedInterval(minVal, maxVal);
}

bool UnreducedInterval::overlaps(const UnreducedInterval &rhs) const {
  return isNotEmpty() && rhs.isNotEmpty() && safeGe(b, rhs.a) && safeLe(a, rhs.b);
}

std::strong_ordering operator<=>(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
  if (safeLt(lhs.a, rhs.a) || (safeEq(lhs.a, rhs.a) && safeLt(lhs.b, rhs.b))) {
    return std::strong_ordering::less;
  }
  if (safeGt(lhs.a, rhs.a) || (safeEq(lhs.a, rhs.a) && safeGt(lhs.b, rhs.b))) {
    return std::strong_ordering::greater;
  }
  return std::strong_ordering::equal;
}

llvm::APSInt UnreducedInterval::width() const {
  llvm::APSInt w;
  if (safeGt(a, b)) {
    // This would be reduced to an empty Interval, so the width is just zero.
    w = llvm::APSInt::getUnsigned(0);
  } else {
    /// Since the range is inclusive, we add one to the difference to get the true width.
    w = expandingSub(b, a)++;
  }
  ensure(safeGe(w, llvm::APSInt::getUnsigned(0)), "cannot have negative width");
  return w;
}

bool UnreducedInterval::isEmpty() const { return safeEq(width(), llvm::APSInt::getUnsigned(0)); }

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
  ensure(
      lhs.getField() == rhs.getField(), "interval operations across differing fields is unsupported"
  );
  const Field &f = lhs.getField();

  // Trivial cases
  if (lhs.isEntire() || rhs.isEntire()) {
    return Interval::Entire(f);
  }
  if (lhs.isEmpty()) {
    return rhs;
  }
  if (rhs.isEmpty()) {
    return lhs;
  }
  if (lhs.isDegenerate() || rhs.isDegenerate()) {
    return lhs.toUnreduced().doUnion(rhs.toUnreduced()).reduce(f);
  }

  // More complex cases
  if (areOneOf<
          {Type::TypeA, Type::TypeA}, {Type::TypeB, Type::TypeB}, {Type::TypeC, Type::TypeC},
          {Type::TypeA, Type::TypeC}, {Type::TypeB, Type::TypeC}>(lhs, rhs)) {
    return Interval(rhs.ty, f, std::min(lhs.a, rhs.a), std::max(lhs.b, rhs.b));
  }
  if (areOneOf<{Type::TypeA, Type::TypeB}>(lhs, rhs)) {
    auto lhsUnred = lhs.firstUnreduced();
    auto opt1 = rhs.firstUnreduced().doUnion(lhsUnred);
    auto opt2 = rhs.secondUnreduced().doUnion(lhsUnred);
    if (opt1.width() <= opt2.width()) {
      return opt1.reduce(f);
    }
    return opt2.reduce(f);
  }
  if (areOneOf<{Type::TypeF, Type::TypeF}, {Type::TypeA, Type::TypeF}>(lhs, rhs)) {
    return lhs.firstUnreduced().doUnion(rhs.firstUnreduced()).reduce(f);
  }
  if (areOneOf<{Type::TypeB, Type::TypeF}>(lhs, rhs)) {
    return lhs.secondUnreduced().doUnion(rhs.firstUnreduced()).reduce(f);
  }
  if (areOneOf<{Type::TypeC, Type::TypeF}>(lhs, rhs)) {
    return Interval::Entire(f);
  }
  if (areOneOf<
          {Type::TypeB, Type::TypeA}, {Type::TypeC, Type::TypeA}, {Type::TypeC, Type::TypeB},
          {Type::TypeF, Type::TypeA}, {Type::TypeF, Type::TypeB}, {Type::TypeF, Type::TypeC}>(
          lhs, rhs
      )) {
    return rhs.join(lhs);
  }
  llvm::report_fatal_error("unhandled join case");
  return Interval::Entire(f);
}

Interval Interval::intersect(const Interval &rhs) const {
  auto &lhs = *this;
  llvm::errs() << __PRETTY_FUNCTION__ << " lhs, rhs " << lhs << ", " << rhs << '\n';
  llvm::errs().indent(4) << lhs.getField().prime() << ", " << rhs.getField().prime() << '\n';
  ensure(
      lhs.getField() == rhs.getField(), "interval operations across differing fields is unsupported"
  );
  llvm::errs().indent(4) << "begin trivial cases\n";
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
  llvm::errs().indent(4) << "end trivial cases\n";

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
  // intersect checks that we're in the same field
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
    if (a != intersection.a && b != intersection.b) {
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
    if (a != intersection.b && b != intersection.a) {
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

FailureOr<Interval> operator/(const Interval &lhs, const Interval &rhs) {
  ensure(lhs.getField() == rhs.getField(), "cannot divide intervals in different fields");
  const auto &field = rhs.getField();
  if (rhs.width() > field.one()) {
    return Interval::Entire(field);
  }
  if (rhs.a.isZero()) {
    return failure();
  }
  return success(UnreducedInterval(lhs.a / rhs.a, lhs.b / rhs.a).reduce(field));
}

Interval operator%(const Interval &lhs, const Interval &rhs) {
  ensure(
      lhs.getField() == rhs.getField(), "interval operations across differing fields is unsupported"
  );
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

} // namespace llzk
