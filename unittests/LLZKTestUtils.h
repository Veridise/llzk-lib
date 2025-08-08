//===-- LLZKTestUtils.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Util/APIntHelper.h"
#include "llzk/Util/Debug.h"

#include <gtest/gtest.h>

/// @brief Check a given condition, outputing an error using the debug::Appender
/// if `cond` is false. Use this wrapper for checks on type `T` where `T` cannot
/// be written directly to std C++ ostreams.
template <typename T>
static testing::AssertionResult checkCond(const T &expected, const T &actual, bool cond) {
  if (cond) {
    return testing::AssertionSuccess();
  }
  std::string errMsg;
  llzk::debug::Appender(errMsg) << "expected " << expected << ", actual is " << actual;
  return testing::AssertionFailure() << errMsg;
}

/// Uses a bitwidth-safe comparison method to check if expected == actual
inline static void AssertSafeEq(const llvm::APSInt &expected, const llvm::APSInt &actual) {
  ASSERT_TRUE(checkCond(expected, actual, llzk::safeEq(expected, actual)));
}
