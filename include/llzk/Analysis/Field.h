//===-- Field.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/SMTAPI.h>

#include <string_view>

namespace llzk {

/// @brief Information about the prime finite field used for the interval analysis.
/// @note Seem implementation of initKnownFields for supported primes.
class Field {
public:
  /// @brief Get a Field from a given field name string.
  /// @param fieldName The name of the field.
  static const Field &getField(const char *fieldName);

  Field() = delete;
  Field(const Field &) = default;
  Field(Field &&) = default;
  Field &operator=(const Field &) = default;

  /// @brief For the prime field p, returns p.
  llvm::APSInt prime() const { return primeMod; }

  /// @brief Returns p / 2.
  llvm::APSInt half() const { return halfPrime; }

  /// @brief Returns i as a field element
  inline llvm::APSInt felt(unsigned i) const { return reduce(i); }

  /// @brief Returns 0 at the bitwidth of the field.
  inline llvm::APSInt zero() const { return felt(0); }

  /// @brief Returns 1 at the bitwidth of the field.
  inline llvm::APSInt one() const { return felt(1); }

  /// @brief Returns p - 1, which is the max value possible in a prime field described by p.
  inline llvm::APSInt maxVal() const { return prime() - one(); }

  /// @brief Returns i mod p and reduces the result into the appropriate bitwidth.
  llvm::APSInt reduce(llvm::APSInt i) const;
  llvm::APSInt reduce(unsigned i) const;

  inline unsigned bitWidth() const { return primeMod.getBitWidth(); }

  /// @brief Create a SMT solver symbol with the current field's bitwidth.
  llvm::SMTExprRef createSymbol(llvm::SMTSolverRef solver, const char *name) const {
    return solver->mkSymbol(name, solver->getBitvectorSort(bitWidth()));
  }

  friend bool operator==(const Field &lhs, const Field &rhs) {
    return lhs.primeMod == rhs.primeMod;
  }

private:
  Field(std::string_view primeStr);
  Field(llvm::APSInt p, llvm::APSInt h) : primeMod(p), halfPrime(h) {}

  llvm::APSInt primeMod, halfPrime;

  static void initKnownFields(llvm::DenseMap<llvm::StringRef, Field> &knownFields);
};

} // namespace llzk
