//===-- APIntHelper.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines helpers for manipulating APInts/APSInts for large numbers
/// and operations over those numbers that may require bit width changes.
/// NOTE: MLIR 19/20 introduces a dynamic version of APInt that manages bitwidths
/// automatically. When we upgrade LLZK to a newer version of MLIR, we can remove
/// these utilities in favor of that.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/StringRef.h>

#include <algorithm>
#include <initializer_list>

namespace llzk {

/// @brief Safely add lhs and rhs, expanding the width of the result as necessary.
/// Numbers are never truncated here, as this assumes that truncation will occur
/// with the reduction to the prime field.
/// @param lhs an N bit number with an allocated width of X
/// @param rhs an M bit number with an allocated width of Y
/// @return A number that is max(N, M) + 1 bits wide with an allocated width of max(max(N, M) + 2,
/// X, Y) bits (+2 to accomodate for the sign bit).
llvm::APSInt expandingAdd(llvm::APSInt lhs, llvm::APSInt rhs);

/// @brief Safely subtract lhs and rhs, expanding the width of the result as necessary.
/// Numbers are never truncated here, as this assumes that truncation will occur
/// with the reduction to the prime field.
/// @param lhs an N bit number with an allocated width of X
/// @param rhs an M bit number with an allocated width of Y
/// @return A number that is max(N, M) + 1 bits wide with an allocated witdh of max(max(N, M) + 2,
/// X, Y) bits (+2 to accomodate for the sign bit).
llvm::APSInt expandingSub(llvm::APSInt lhs, llvm::APSInt rhs);

/// @brief Safely multiple lhs and rhs, expanding the width of the result as necessary.
/// Numbers are never truncated here, as this assumes that truncation will occur
/// with the reduction to the prime field.
/// @param lhs an N bit number with an allocated width of X
/// @param rhs an M bit number with an allocated width of Y
/// @return A number that is N + M bits wide with an allocated width of max(N + M +1 , X, Y) bits
/// (+1 to accomodate for the sign bit).
llvm::APSInt expandingMul(llvm::APSInt lhs, llvm::APSInt rhs);

/// @brief Compares lhs and rhs, regardless of the bitwidth of lhs and rhs.
/// @return lhs is less, equal, or greater than rhs
std::strong_ordering safeCmp(llvm::APSInt lhs, llvm::APSInt rhs);

/// @brief Safely converts the given int to a signed int if it is an unsigned int
/// by adding an extra bit for the sign.
llvm::APSInt safeToSigned(llvm::APSInt i);
inline llvm::APSInt safeToSigned(llvm::APInt i) { return safeToSigned(llvm::APSInt(i)); }
inline llvm::APSInt safeToSigned(llvm::StringRef s) { return safeToSigned(llvm::APSInt(s)); }

inline bool safeLt(const llvm::APSInt &lhs, const llvm::APSInt &rhs) {
  return std::is_lt(safeCmp(lhs, rhs));
}

inline bool safeLe(const llvm::APSInt &lhs, const llvm::APSInt &rhs) {
  return std::is_lteq(safeCmp(lhs, rhs));
}

inline bool safeEq(const llvm::APSInt &lhs, const llvm::APSInt &rhs) {
  return std::is_eq(safeCmp(lhs, rhs));
}

inline bool safeNe(const llvm::APSInt &lhs, const llvm::APSInt &rhs) {
  return std::is_neq(safeCmp(lhs, rhs));
}

inline bool safeGt(const llvm::APSInt &lhs, const llvm::APSInt &rhs) {
  return std::is_gt(safeCmp(lhs, rhs));
}

inline bool safeGe(const llvm::APSInt &lhs, const llvm::APSInt &rhs) {
  return std::is_gteq(safeCmp(lhs, rhs));
}

inline llvm::APSInt safeMin(const llvm::APSInt &lhs, const llvm::APSInt &rhs) {
  return std::min(lhs, rhs, safeLt);
}

inline llvm::APSInt safeMin(std::initializer_list<llvm::APSInt> ilist) {
  return std::min(ilist, safeLt);
}

inline llvm::APSInt safeMax(const llvm::APSInt &lhs, const llvm::APSInt &rhs) {
  return std::max(lhs, rhs, safeLt);
}

inline llvm::APSInt safeMax(std::initializer_list<llvm::APSInt> ilist) {
  return std::max(ilist, safeLt);
}

} // namespace llzk
