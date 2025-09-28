//===-- DynamicAPIntTests.cpp - Tests for DynamicAPInt helpers --*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Util/Debug.h"

#include <llvm/Support/raw_ostream.h>

#include <gtest/gtest.h>
#include <string>

#include "../LLZKTestUtils.h"

using namespace llvm;
using namespace llzk;
using namespace std;

static DynamicAPInt bn254 =
    toDynamicAPInt("21888242871839275222246405745257275088696311157297823662689037894645226208583");

struct DynamicAPIntUnaryTests : public testing::Test,
                                public testing::WithParamInterface<DynamicAPInt> {
  static const std::vector<DynamicAPInt> &TestingValues() {
    static std::vector<DynamicAPInt> vals = {
        DynamicAPInt(-1), DynamicAPInt(0), DynamicAPInt(1234), bn254, -1 * bn254,
    };
    return vals;
  }
};

struct DynamicAPIntBinaryTests
    : public testing::Test,
      public testing::WithParamInterface<std::pair<DynamicAPInt, DynamicAPInt>> {
  static const std::vector<std::pair<DynamicAPInt, DynamicAPInt>> &TestingValues() {
    static std::vector<std::pair<DynamicAPInt, DynamicAPInt>> vals = {
        {DynamicAPInt(-1), DynamicAPInt(0)},
        {DynamicAPInt(-1), bn254},
        {DynamicAPInt(0xcafe), DynamicAPInt(0xdeadbeef)}
    };
    return vals;
  }
};

struct DynamicAPIntShiftTests
    : public testing::Test,
      public testing::WithParamInterface<std::pair<DynamicAPInt, unsigned>> {
  static const std::vector<std::pair<DynamicAPInt, unsigned>> &TestingValues() {
    static std::vector<std::pair<DynamicAPInt, unsigned>> vals = {
        {DynamicAPInt(-1), 0},
        {bn254, 0},
        {bn254, 32},
        {DynamicAPInt(100), 32},
    };
    return vals;
  }
};

TEST_P(DynamicAPIntUnaryTests, Conversions) {
  DynamicAPInt p = GetParam();
  DynamicAPInt convert = toDynamicAPInt(toAPSInt(p));
  ASSERT_EQ(p, convert);
}

INSTANTIATE_TEST_SUITE_P(
    Conversions, DynamicAPIntUnaryTests, testing::ValuesIn(DynamicAPIntUnaryTests::TestingValues())
);

static void extendAPSInts(APSInt &a, APSInt &b) {
  unsigned maxBitwidth = max(a.getBitWidth(), b.getBitWidth());
  a = a.extend(maxBitwidth);
  b = b.extend(maxBitwidth);
}

TEST_P(DynamicAPIntBinaryTests, BitAnd) {
  auto [a, b] = GetParam();
  // Commutative
  ASSERT_EQ(a & b, b & a);
  // Equivalent to APSInt operator
  APSInt sa = toAPSInt(a), sb = toAPSInt(b);
  extendAPSInts(sa, sb);
  DynamicAPInt baseline = toDynamicAPInt(sa & sb);
  ASSERT_EQ(a & b, baseline);
}

INSTANTIATE_TEST_SUITE_P(
    BitAnd, DynamicAPIntBinaryTests, testing::ValuesIn(DynamicAPIntBinaryTests::TestingValues())
);

TEST_P(DynamicAPIntBinaryTests, BitOr) {
  auto [a, b] = GetParam();
  // Commutative
  ASSERT_EQ(a | b, b | a);
  // Equivalent to APSInt operator
  APSInt sa = toAPSInt(a), sb = toAPSInt(b);
  extendAPSInts(sa, sb);
  ASSERT_EQ(a | b, toDynamicAPInt(sa | sb));
}

INSTANTIATE_TEST_SUITE_P(
    bidirectional_iterator_tag, DynamicAPIntBinaryTests,
    testing::ValuesIn(DynamicAPIntBinaryTests::TestingValues())
);

TEST_P(DynamicAPIntBinaryTests, BitXor) {
  auto [a, b] = GetParam();
  // Commutative
  ASSERT_EQ(a ^ b, b ^ a);
  // Equivalent to APSInt operator
  APSInt sa = toAPSInt(a), sb = toAPSInt(b);
  extendAPSInts(sa, sb);
  ASSERT_EQ(a ^ b, toDynamicAPInt(sa ^ sb));
}

INSTANTIATE_TEST_SUITE_P(
    BitXor, DynamicAPIntBinaryTests, testing::ValuesIn(DynamicAPIntBinaryTests::TestingValues())
);

TEST_P(DynamicAPIntShiftTests, ShiftLeft) {
  auto [a, b] = GetParam();
  // Equivalent to APSInt operator
  APSInt base = toAPSInt(a);
  base = base.extend(base.getBitWidth() + b);

  ASSERT_EQ(a << toDynamicAPInt(APSInt::get(b)), toDynamicAPInt(base << b));
}

INSTANTIATE_TEST_SUITE_P(
    ShiftLeft, DynamicAPIntShiftTests, testing::ValuesIn(DynamicAPIntShiftTests::TestingValues())
);

TEST_P(DynamicAPIntShiftTests, ShiftRight) {
  auto [a, b] = GetParam();
  // Equivalent to APSInt operator
  ASSERT_EQ(a >> toDynamicAPInt(APSInt::get(b)), toDynamicAPInt(toAPSInt(a) >> b));
}

INSTANTIATE_TEST_SUITE_P(
    ShiftRight, DynamicAPIntShiftTests, testing::ValuesIn(DynamicAPIntShiftTests::TestingValues())
);
