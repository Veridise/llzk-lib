//===-- IntervalTests.cpp - Unit tests for interval analysis ----*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/Intervals.h"
#include "llzk/Util/Debug.h"

#include <gtest/gtest.h>
#include <string>

#include "../LLZKTestUtils.h"

using namespace llzk;

class IntervalTests : public testing::Test {
protected:
  const Field &f;
  const Interval empty, entire;

  IntervalTests()
      : f(Field::getField("babybear")), empty(Interval::Empty(f)), entire(Interval::Entire(f)) {}

  inline static void
  AssertUnreducedIntervalEq(const UnreducedInterval &expected, const UnreducedInterval &actual) {
    ASSERT_TRUE(checkCond(expected, actual, expected == actual));
  }

  inline static void AssertIntervalEq(const Interval &expected, const Interval &actual) {
    ASSERT_TRUE(checkCond(expected, actual, expected == actual));
  }
};

TEST_F(IntervalTests, UnreducedIntervalOverlap) {
  UnreducedInterval a(0, 100), b(100, 200), c(101, 300), d(1, 0);
  ASSERT_TRUE(a.overlaps(b));
  ASSERT_TRUE(b.overlaps(a));
  ASSERT_FALSE(a.overlaps(c));
  ASSERT_TRUE(b.overlaps(c));
  ASSERT_FALSE(d.overlaps(a));
}

TEST_F(IntervalTests, UnreducedIntervalWidth) {
  // Standard width.
  UnreducedInterval a(0, 100);
  ASSERT_EQ(f.felt(101), a.width());
  // Standard width for a single element range.
  UnreducedInterval b(4, 4);
  ASSERT_EQ(f.one(), b.width());
  // Range of this will be 0 since a > b.
  UnreducedInterval c(4, 3);
  ASSERT_EQ(f.zero(), c.width());
}

TEST_F(IntervalTests, IntervalWidth) {
  // Standard width.
  Interval a = UnreducedInterval(0, 100).reduce(f);
  ASSERT_EQ(f.felt(101), a.width());
  // Standard width for a single element range.
  Interval b = UnreducedInterval(4, 4).reduce(f);
  ASSERT_EQ(f.one(), b.width());
  // Range of this will be 0 since a > b.
  Interval c = UnreducedInterval(4, 3).reduce(f);
  ASSERT_EQ(f.zero(), c.width());

  ASSERT_EQ(Interval::Entire(f).width(), f.prime());
  ASSERT_EQ(Interval::Empty(f).width(), f.zero());
  ASSERT_EQ(Interval::Degenerate(f, f.felt(7)).width(), f.one());
}

TEST_F(IntervalTests, Partitions) {
  UnreducedInterval a(0, 100), b(100, 200), c(101, 300), d(1, 0), s1(1, 10), s2(3, 7);

  // Some basic overlapping intervals
  AssertUnreducedIntervalEq(a, a.computeLTPart(b));
  AssertUnreducedIntervalEq(a, a.computeLEPart(b));
  AssertUnreducedIntervalEq(b, b.computeGEPart(a));
  AssertUnreducedIntervalEq(b, b.computeGTPart(a));

  AssertUnreducedIntervalEq(UnreducedInterval(1, 6), s1.computeLTPart(s2));
  AssertUnreducedIntervalEq(UnreducedInterval(1, 7), s1.computeLEPart(s2));
  AssertUnreducedIntervalEq(UnreducedInterval(4, 10), s1.computeGTPart(s2));
  AssertUnreducedIntervalEq(UnreducedInterval(3, 10), s1.computeGEPart(s2));

  // Some non-overlapping intervals, should all be empty
  ASSERT_TRUE(b.computeLTPart(a).reduce(f).isEmpty());
  ASSERT_TRUE(a.computeGTPart(b).reduce(f).isEmpty());
  ASSERT_TRUE(c.computeLEPart(a).reduce(f).isEmpty());
  ASSERT_TRUE(a.computeGEPart(c).reduce(f).isEmpty());

  // Any computation where LHS or RHS is empty returns LHS.
  AssertUnreducedIntervalEq(a, a.computeLTPart(d));
  AssertUnreducedIntervalEq(b, b.computeLEPart(d));
  AssertUnreducedIntervalEq(c, c.computeGTPart(d));
  AssertUnreducedIntervalEq(d, d.computeGEPart(d));
  AssertUnreducedIntervalEq(d, d.computeLTPart(a));
  AssertUnreducedIntervalEq(d, d.computeLEPart(b));
  AssertUnreducedIntervalEq(d, d.computeGTPart(c));
  AssertUnreducedIntervalEq(d, d.computeGEPart(d));
}

TEST_F(IntervalTests, Difference) {
  // Following the examples in the Interval::difference docs.
  auto a = Interval::TypeA(f, f.felt(1), f.felt(10));
  auto b = Interval::TypeA(f, f.felt(5), f.felt(11));
  auto c = Interval::TypeA(f, f.felt(5), f.felt(6));

  ASSERT_EQ(Interval::TypeA(f, f.felt(1), f.felt(4)), a.difference(b));
  ASSERT_EQ(a, a.difference(c));
}

TEST_F(IntervalTests, UnreduceReduce) {
  // unreducing and reducing should not be destructive
  AssertIntervalEq(Interval::Entire(f), Interval::Entire(f).toUnreduced().reduce(f));
  AssertIntervalEq(Interval::Empty(f), Interval::Empty(f).toUnreduced().reduce(f));
  AssertIntervalEq(
      Interval::Degenerate(f, f.felt(8)), Interval::Degenerate(f, f.felt(8)).toUnreduced().reduce(f)
  );
}

TEST_F(IntervalTests, AdditiveIdentities) {
  // Empty + Empty = Empty
  AssertIntervalEq(empty, empty + empty);
  // Entire + Entire = Entire
  AssertIntervalEq(entire, entire + entire);
  // Entire + Empty = Entire
  AssertIntervalEq(entire, entire + empty);
  AssertIntervalEq(entire, empty + entire);
}

TEST_F(IntervalTests, NegativeIdentities) {
  // negative "entire" should still be "entire"
  AssertIntervalEq(Interval::Entire(f), -Interval::Entire(f));

  // negative "empty" should still be "empty"
  AssertIntervalEq(Interval::Empty(f), -Interval::Empty(f));

  // -1 should be max value when reduced (1 + (-1) % p == 1 + (p - 1) % p == p % p == 0)
  auto maxValDegen = Interval::Degenerate(f, f.maxVal());
  auto oneDegen = Interval::Degenerate(f, f.one());
  AssertIntervalEq(maxValDegen, -oneDegen);
}

TEST_F(IntervalTests, BitwiseNot) {
  auto one = Interval::Degenerate(f, f.one());
  auto a = Interval::TypeA(f, f.zero(), f.felt(7));
  auto notA = Interval::TypeF(f, f.prime() - f.felt(6), f.one());
  AssertIntervalEq(~a, one - a);
  AssertIntervalEq(notA, ~a);
}
