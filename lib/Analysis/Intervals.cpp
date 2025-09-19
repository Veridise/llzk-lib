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
  const auto &lhs = *this;
  return UnreducedInterval(safeMax(lhs.a, rhs.a), safeMin(lhs.b, rhs.b));
}

UnreducedInterval UnreducedInterval::doUnion(const UnreducedInterval &rhs) const {
  const auto &lhs = *this;
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

UnreducedInterval UnreducedInterval::operator-() const {
  if (isEmpty()) {
    return *this;
  }
  return UnreducedInterval(-b, -a);
}

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
    // Since the range is inclusive, we add one to the difference to get the true width.
    w = expandingSub(b, a)++;
  }
  ensure(safeGe(w, llvm::APSInt::getUnsigned(0)), "cannot have negative width");
  return w;
}

bool UnreducedInterval::isEmpty() const { return safeEq(width(), llvm::APSInt::getUnsigned(0)); }

/* Interval */

const Field &checkFields(const Interval &lhs, const Interval &rhs) {
  ensure(
      lhs.getField() == rhs.getField(), "interval operations across differing fields is unsupported"
  );
  return lhs.getField();
}

UnreducedInterval Interval::toUnreduced() const {
  if (isEmpty()) {
    // Since ranges are inclusive, empty is encoded as `[a, b]` where `a` > `b`.
    // This matches the definition provided by UnreducedInterval::width().
    return UnreducedInterval(field.get().one(), field.get().zero());
  }
  if (isEntire()) {
    return UnreducedInterval(field.get().zero(), field.get().maxVal());
  }
  return UnreducedInterval(a, b);
}

UnreducedInterval Interval::firstUnreduced() const {
  if (is<Type::TypeF>()) {
    return UnreducedInterval(a - field.get().prime(), b);
  }
  return toUnreduced();
}

UnreducedInterval Interval::secondUnreduced() const {
  ensure(is<Type::TypeA, Type::TypeB, Type::TypeC>(), "unsupported range type");
  return UnreducedInterval(a - field.get().prime(), b - field.get().prime());
}

Interval Interval::join(const Interval &rhs) const {
  const auto &lhs = *this;
  const Field &f = checkFields(lhs, rhs);

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
  const auto &lhs = *this;
  const Field &f = checkFields(lhs, rhs);
  // Trivial cases
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return Interval::Empty(f);
  }
  if (lhs.isEntire()) {
    return rhs;
  }
  if (rhs.isEntire()) {
    return lhs;
  }
  if (lhs.isDegenerate() || rhs.isDegenerate()) {
    return lhs.toUnreduced().intersect(rhs.toUnreduced()).reduce(f);
  }

  // More complex cases
  if (areOneOf<
          {Type::TypeA, Type::TypeA}, {Type::TypeB, Type::TypeB}, {Type::TypeC, Type::TypeC},
          {Type::TypeA, Type::TypeC}, {Type::TypeB, Type::TypeC}>(lhs, rhs)) {
    auto maxA = std::max(lhs.a, rhs.a);
    auto minB = std::min(lhs.b, rhs.b);
    if (maxA <= minB) {
      return Interval(lhs.ty, f, maxA, minB);
    } else {
      return Interval::Empty(f);
    }
  }
  if (areOneOf<{Type::TypeA, Type::TypeB}>(lhs, rhs)) {
    return Interval::Empty(f);
  }
  if (areOneOf<{Type::TypeF, Type::TypeF}, {Type::TypeA, Type::TypeF}>(lhs, rhs)) {
    return lhs.firstUnreduced().intersect(rhs.firstUnreduced()).reduce(f);
  }
  if (areOneOf<{Type::TypeB, Type::TypeF}>(lhs, rhs)) {
    return lhs.secondUnreduced().intersect(rhs.firstUnreduced()).reduce(f);
  }
  if (areOneOf<{Type::TypeC, Type::TypeF}>(lhs, rhs)) {
    auto rhsUnred = rhs.firstUnreduced();
    auto opt1 = lhs.firstUnreduced().intersect(rhsUnred).reduce(f);
    auto opt2 = lhs.secondUnreduced().intersect(rhsUnred).reduce(f);
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
  return Interval::Empty(f);
}

Interval Interval::difference(const Interval &other) const {
  const Field &f = checkFields(*this, other);
  // intersect checks that we're in the same field
  Interval intersection = intersect(other);
  if (intersection.isEmpty()) {
    // There's nothing to remove, so just return this
    return *this;
  }

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

Interval Interval::operator~() const {
  return Interval::Degenerate(field.get(), field.get().one()) - *this;
}

Interval operator+(const Interval &lhs, const Interval &rhs) {
  const Field &f = checkFields(lhs, rhs);
  if (lhs.isEmpty()) {
    return rhs;
  }
  if (rhs.isEmpty()) {
    return lhs;
  }
  return (lhs.firstUnreduced() + rhs.firstUnreduced()).reduce(f);
}

Interval operator-(const Interval &lhs, const Interval &rhs) { return lhs + (-rhs); }

Interval operator*(const Interval &lhs, const Interval &rhs) {
  const Field &f = checkFields(lhs, rhs);
  auto zeroInterval = Interval::Degenerate(f, f.zero());
  if (lhs == zeroInterval || rhs == zeroInterval) {
    return zeroInterval;
  }
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return Interval::Empty(f);
  }
  if (lhs.isEntire() || rhs.isEntire()) {
    return Interval::Entire(f);
  }

  if (Interval::areOneOf<{Interval::Type::TypeB, Interval::Type::TypeB}>(lhs, rhs)) {
    return (lhs.secondUnreduced() * rhs.secondUnreduced()).reduce(f);
  }
  return (lhs.firstUnreduced() * rhs.firstUnreduced()).reduce(f);
}

FailureOr<Interval> operator/(const Interval &lhs, const Interval &rhs) {
  const Field &f = checkFields(lhs, rhs);
  if (rhs.width() > f.one()) {
    return Interval::Entire(f);
  }
  if (rhs.a.isZero()) {
    return failure();
  }
  return success(UnreducedInterval(lhs.a / rhs.a, lhs.b / rhs.a).reduce(f));
}

Interval operator%(const Interval &lhs, const Interval &rhs) {
  const Field &f = checkFields(lhs, rhs);
  return UnreducedInterval(f.zero(), rhs.b).reduce(f);
}

Interval operator&(const Interval &lhs, const Interval &rhs) {
  const Field &f = checkFields(lhs, rhs);
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return Interval::Empty(f);
  }
  if (lhs.isDegenerate() && rhs.isDegenerate()) {
    return Interval::Degenerate(f, lhs.a & rhs.a);
  } else if (lhs.isDegenerate()) {
    return UnreducedInterval(f.zero(), lhs.a).reduce(f);
  } else if (rhs.isDegenerate()) {
    return UnreducedInterval(f.zero(), rhs.a).reduce(f);
  }
  return Interval::Entire(f);
}

Interval operator<<(const Interval &lhs, const Interval &rhs) {
  const Field &f = checkFields(lhs, rhs);
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return Interval::Empty(f);
  }
  if (lhs.isDegenerate() && rhs.isDegenerate()) {
    if (safeGt(rhs.a, APSInt::getUnsigned(f.bitWidth()))) {
      return Interval::Entire(f);
    }

    unsigned shiftAmt = rhs.a.getZExtValue();
    auto v = lhs.a.relativeShl(shiftAmt);
    return UnreducedInterval(v, v).reduce(f);
  }
  return Interval::Entire(f);
}

Interval operator>>(const Interval &lhs, const Interval &rhs) {
  const Field &f = checkFields(lhs, rhs);
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return Interval::Empty(f);
  }
  if (lhs.isDegenerate() && rhs.isDegenerate()) {
    if (safeGt(rhs.a, APSInt::getUnsigned(f.bitWidth()))) {
      return Interval::Degenerate(f, f.zero());
    }

    unsigned shiftAmt = rhs.a.getZExtValue();
    return Interval::Degenerate(f, lhs.a.relativeShr(shiftAmt));
  }
  return Interval::Entire(f);
}

llvm::APSInt Interval::width() const {
  switch (ty) {
  case Type::Empty:
    return field.get().zero();
  case Type::Degenerate:
    return field.get().one();
  case Type::Entire:
    return field.get().prime();
  default:
    return field.get().reduce(toUnreduced().width());
  }
}

Interval boolAnd(const Interval &lhs, const Interval &rhs) {
  ensure(
      lhs.getField() == rhs.getField(), "interval operations across differing fields is unsupported"
  );
  ensure(lhs.isBoolean() && rhs.isBoolean(), "operation only supported for boolean-type intervals");
  const auto &field = rhs.getField();

  if (lhs.isBoolFalse() || rhs.isBoolFalse()) {
    return Interval::False(field);
  }
  if (lhs.isBoolTrue() && rhs.isBoolTrue()) {
    return Interval::True(field);
  }

  return Interval::Boolean(field);
}

Interval boolOr(const Interval &lhs, const Interval &rhs) {
  ensure(
      lhs.getField() == rhs.getField(), "interval operations across differing fields is unsupported"
  );
  ensure(lhs.isBoolean() && rhs.isBoolean(), "operation only supported for boolean-type intervals");
  const auto &field = rhs.getField();

  if (lhs.isBoolFalse() && rhs.isBoolFalse()) {
    return Interval::False(field);
  }
  if (lhs.isBoolTrue() || rhs.isBoolTrue()) {
    return Interval::True(field);
  }

  return Interval::Boolean(field);
}

Interval boolXor(const Interval &lhs, const Interval &rhs) {
  ensure(
      lhs.getField() == rhs.getField(), "interval operations across differing fields is unsupported"
  );
  ensure(lhs.isBoolean() && rhs.isBoolean(), "operation only supported for boolean-type intervals");
  const auto &field = rhs.getField();

  // Xor-ing anything with [0, 1] could still result in either case, so just return
  // the full boolean range.
  if (lhs.isBoolEither() || rhs.isBoolEither()) {
    return Interval::Boolean(lhs.getField());
  }

  if (lhs.isBoolTrue() && rhs.isBoolTrue()) {
    return Interval::False(field);
  }
  if (lhs.isBoolTrue() || rhs.isBoolTrue()) {
    return Interval::True(field);
  }
  if (lhs.isBoolFalse() && rhs.isBoolFalse()) {
    return Interval::False(field);
  }

  return Interval::Boolean(field);
}

Interval boolNot(const Interval &iv) {
  ensure(iv.isBoolean(), "operation only supported for boolean-type intervals");
  const auto &field = iv.getField();

  if (iv.isBoolTrue()) {
    return Interval::False(field);
  }
  if (iv.isBoolFalse()) {
    return Interval::True(field);
  }

  return iv;
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
