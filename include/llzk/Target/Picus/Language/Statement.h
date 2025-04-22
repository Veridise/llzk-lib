//===-- Statement.h - Picus statements definition ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines classes for handling Picus statements.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <llzk/Target/Picus/Language/Expression.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

#include <memory>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace picus {

/// A root expression in Picus that does not yield a value.
class Statement {
public:
  virtual ~Statement() = default;
  virtual void print(llvm::raw_ostream &) const = 0;
};

/// Declares a variable as an input of a program.
class InputStmt : public Statement {
public:
  InputStmt(VarExpr);
  void print(llvm::raw_ostream &) const override;

private:
  VarExpr var;
};

/// Declares a variable as an output of a program.
class OutputStmt : public Statement {
public:
  OutputStmt(VarExpr);
  void print(llvm::raw_ostream &) const override;

private:
  VarExpr var;
};

/// Declares an assertion that needs to be safisfied.
class AssertStmt : public Statement {
public:
  AssertStmt(Expression::ptr);
  void print(llvm::raw_ostream &) const override;

private:
  Expression::ptr expr;
};

/// Declares a lookup in the circuit.
class LookupStmt : public Statement {
public:
  LookupStmt(Expression::ptr, Expression::ptr);
  LookupStmt(llvm::MutableArrayRef<Expression::ptr>, llvm::MutableArrayRef<Expression::ptr>);
  void print(llvm::raw_ostream &) const override;

private:
  std::vector<Expression::ptr> exprs, fixedRefs;
};

struct CallStmtData {
  llvm::StringRef moduleName;
  llvm::ArrayRef<llvm::StringRef> inputs, outputs;
};

/// Declares a call to another module.
class CallStmt : public Statement {
public:
  CallStmt(CallStmtData);
  void print(llvm::raw_ostream &) const override;

private:
  llvm::StringRef moduleName;
  llvm::SmallVector<llvm::StringRef> inputs, outputs;
};

} // namespace picus

namespace llvm {
raw_ostream &operator<<(raw_ostream &, const picus::Statement &);
raw_ostream &operator<<(raw_ostream &, const std::unique_ptr<picus::Statement> &);

// These 3 may not be needed anymore.
raw_ostream &operator<<(raw_ostream &, const picus::InputStmt &);
raw_ostream &operator<<(raw_ostream &, const picus::OutputStmt &);
raw_ostream &operator<<(raw_ostream &, const picus::AssertStmt &);
} // namespace llvm
