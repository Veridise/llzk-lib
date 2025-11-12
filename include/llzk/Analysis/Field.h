//===-- Field.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Util/DynamicAPIntHelper.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/Support/SMTAPI.h>

#include <string_view>

namespace llzk {

/// @brief Information about the prime finite field used for the interval analysis.
/// @note See implementation of initKnownFields for supported primes.
/// @note We use DynamicAPInt to support arithmetic that may require increasing
/// or signed arithmetic (e.g., multiplying field elements before applying the
/// modulus).
class Field {
public:
  /// @brief Get a Field from a given field name string.
  /// @param fieldName The name of the field.
  static const Field &getField(const char *fieldName);

  Field() = delete;
  Field(const Field &) = default;
  Field(Field &&) noexcept = default;
  Field &operator=(const Field &) = default;

  /// @brief For the prime field p, returns p.
  llvm::DynamicAPInt prime() const { return primeMod; }

  /// @brief Returns p / 2.
  llvm::DynamicAPInt half() const { return halfPrime; }

  /// @brief Returns i as a signed field element
  inline llvm::DynamicAPInt felt(int i) const { return reduce(i); }

  /// @brief Returns 0 at the bitwidth of the field.
  inline llvm::DynamicAPInt zero() const { return felt(0); }

  /// @brief Returns 1 at the bitwidth of the field.
  inline llvm::DynamicAPInt one() const { return felt(1); }

  /// @brief Returns p - 1, which is the max value possible in a prime field described by p.
  inline llvm::DynamicAPInt maxVal() const { return prime() - one(); }

  /// @brief Returns the multiplicative inverse of `i` in prime field `p`.
  llvm::DynamicAPInt inv(const llvm::DynamicAPInt &i) const;

  llvm::DynamicAPInt inv(const llvm::APInt &i) const;

  /// @brief Returns i mod p and reduces the result into the appropriate bitwidth.
  /// Field elements are returned as signed integers so that negation functions
  /// as expected (i.e., reducing -1 will yield p-1).
  llvm::DynamicAPInt reduce(const llvm::DynamicAPInt &i) const;
  inline llvm::DynamicAPInt reduce(int i) const { return reduce(llvm::DynamicAPInt(i)); }
  llvm::DynamicAPInt reduce(const llvm::APInt &i) const;

  inline unsigned bitWidth() const { return bitwidth; }

  /// @brief Create a SMT solver symbol with the current field's bitwidth.
  llvm::SMTExprRef createSymbol(llvm::SMTSolverRef solver, const char *name) const {
    return solver->mkSymbol(name, solver->getBitvectorSort(bitWidth()));
  }

  friend bool operator==(const Field &lhs, const Field &rhs) {
    return lhs.primeMod == rhs.primeMod;
  }

private:
  Field(std::string_view primeStr);

  llvm::DynamicAPInt primeMod, halfPrime;
  unsigned bitwidth;

  static void initKnownFields(llvm::DenseMap<llvm::StringRef, Field> &knownFields);
};

} // namespace llzk
