#include "llzk/Dialect/LLZK/Analysis/IntervalAnalysis.h"
#include "llzk/Dialect/LLZK/Util/Debug.h"

#include <gtest/gtest.h>
#include <string>

using namespace llzk;

class IntervalTests : public testing::Test {
protected:
  const Field &f;

  IntervalTests() : f(Field::getField("babybear")) {}

  /// Uses a bitwidth-safe comparison method to check if expected == actual
  static testing::AssertionResult
  checkSafeEq(const llvm::APSInt &expected, const llvm::APSInt &actual) {
    if (safeEq(expected, actual)) {
      return testing::AssertionSuccess();
    }
    std::string errMsg;
    debug::Appender(errMsg) << "expected " << expected << ", actual is " << actual;
    return testing::AssertionFailure() << errMsg;
  }

  inline static void AssertSafeEq(const llvm::APSInt &expected, const llvm::APSInt &actual) {
    ASSERT_TRUE(checkSafeEq(expected, actual));
  }

  inline static void
  AssertUnreducedIntervalsEq(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
    ASSERT_TRUE(lhs == rhs);
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
  AssertSafeEq(f.felt(101), a.width());
  // Standard width for a single element range.
  UnreducedInterval b(4, 4);
  AssertSafeEq(f.one(), b.width());
  // Range of this will be 0 since a > b.
  UnreducedInterval c(4, 3);
  AssertSafeEq(f.zero(), c.width());
}

TEST_F(IntervalTests, Partitions) {
  UnreducedInterval a(0, 100), b(100, 200), c(101, 300), d(1, 0);

  // Some basic overlaping intervals
  AssertUnreducedIntervalsEq(UnreducedInterval(0, 99), a.computeLTPart(b));
  AssertUnreducedIntervalsEq(UnreducedInterval(0, 100), a.computeLEPart(b));
  AssertUnreducedIntervalsEq(a.computeLTPart(b), b.computeGEPart(a));
  ASAssertUnreducedIntervalsEqSERT_EQ(a.computeLEPart(b), b.computeGTPart(a));

  // Some non-overlaping intervals, should all be empty
  ASSERT_TRUE(b.computeLTPart(a).reduce(f).isEmpty());
  ASSERT_TRUE(a.computeGTPart(b).reduce(f).isEmpty());
  ASSERT_TRUE(c.computeLEPart(a).reduce(f).isEmpty());
  ASSERT_TRUE(a.computeGEPart(c).reduce(f).isEmpty());

  // Any computation where LHS or RHS is empty returns LHS.
  AssertUnreducedIntervalsEq(a, a.computeLTPart(d));
  AssertUnreducedIntervalsEq(b, b.computeLEPart(d));
  AssertUnreducedIntervalsEq(c, c.computeGTPart(d));
  AssertUnreducedIntervalsEq(d, d.computeGEPart(d));
  AssertUnreducedIntervalsEq(d, d.computeLTPart(a));
  AssertUnreducedIntervalsEq(d, d.computeLEPart(b));
  AssertUnreducedIntervalsEq(d, d.computeGTPart(c));
  AssertUnreducedIntervalsEq(d, d.computeGEPart(d));
}
