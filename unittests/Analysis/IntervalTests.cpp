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
