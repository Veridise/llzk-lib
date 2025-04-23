//===-- Circuit.h - Picus language top-level definitions --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines a class for handling Picus circuits.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <llzk/Target/Picus/Language/Expression.h>
#include <llzk/Target/Picus/Language/Statement.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>

namespace llvm {
class raw_ostream;
}

namespace picus {

/// Represents a collection of constant values used for performing lookups.
/// They can be referenced by a given name. That name is lost when generating
/// the lisp Picus code.
class FixedValues {
public:
  void print(llvm::raw_ostream &) const;

  void addFixedValues(llvm::StringRef, llvm::ArrayRef<ConstExpr>);
  void addFixedValues(llvm::StringRef, ConstExpr);

  Expression::ptr getFixedValueRef(llvm::StringRef);

private:
  llvm::StringMap<llvm::SmallVector<ConstExpr>> values;
};

/// Logical division of statements. A LLZK struct gets translated into a module.
class Module {
public:
  Module(llvm::StringRef);
  void print(llvm::raw_ostream &) const;

  void addStatement(std::unique_ptr<Statement> stmt) { statements.push_back(std::move(stmt)); }

private:
  llvm::StringRef name;
  llvm::SmallVector<std::unique_ptr<Statement>> statements;
};

/// Declares the prime number used by the circuit.
class PrimeNumber {
public:
  PrimeNumber(llvm::APInt);
  void print(llvm::raw_ostream &) const;

private:
  llvm::APInt prime;
};

/// Top level class that represents a complete Picus program.
/// Comprised of a sequence of top-level statements defining elements of the circuit.
class Circuit {
public:
  Circuit(PrimeNumber);
  /// Prints the program as an collection of s-expressions into the output stream.
  void print(llvm::raw_ostream &) const;

  /// Creates a new module and returns a reference to it.
  Module &emplaceModule(llvm::StringRef name);

  FixedValues &getFixedValues() { return fixed; }

private:
  PrimeNumber prime;
  llvm::StringMap<Module> modules;
  FixedValues fixed;
};

} // namespace picus
