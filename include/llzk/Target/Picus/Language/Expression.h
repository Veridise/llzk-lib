//===-- Expression.h - Picus language expressions ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines classes for handling Picus expressions.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/StringRef.h>

#include <memory>

namespace llvm {
class raw_ostream;
}

namespace picus {

/// A expression in Picus.
class Expression {
public:
  using ptr = std::unique_ptr<Expression>;

  virtual ~Expression() = default;
  virtual void print(llvm::raw_ostream &) const = 0;
};

/// A expression symbolized by a variable.
class VarExpr : public Expression {
public:
  VarExpr(llvm::StringRef);
  void print(llvm::raw_ostream &) const override;

private:
  llvm::StringRef name;
};

/// A constant valued expression.
class ConstExpr : public Expression {
public:
  ConstExpr(llvm::APInt);
  void print(llvm::raw_ostream &) const override;

private:
  llvm::APInt value;
};

/// A reference to a fixed value.
class FixedValueRef : public Expression {
public:
  FixedValueRef(unsigned);
  void print(llvm::raw_ostream &) const override;

private:
  unsigned id;
};

/// Base class for any binary expression.
class BinaryExpr : public Expression {
public:
  BinaryExpr(Expression::ptr, Expression::ptr);
  void print(llvm::raw_ostream &) const override;

protected:
  virtual constexpr llvm::StringRef op() const = 0;

private:
  Expression::ptr lhs, rhs;
};

/// Base class for any unary expression.
class UnaryExpr : public Expression {
public:
  UnaryExpr(Expression::ptr);
  void print(llvm::raw_ostream &) const override;

protected:
  virtual constexpr llvm::StringRef op() const = 0;

private:
  Expression::ptr value;
};

/// Addition expression.
class AddExpr : public BinaryExpr {
public:
  using BinaryExpr::BinaryExpr;
  constexpr llvm::StringRef op() const override { return "+"; }
};

/// Subtraction expression.
class SubExpr : public BinaryExpr {
public:
  using BinaryExpr::BinaryExpr;
  constexpr llvm::StringRef op() const override { return "-"; }
};

/// Multiplication expression.
class MulExpr : public BinaryExpr {
public:
  using BinaryExpr::BinaryExpr;
  constexpr llvm::StringRef op() const override { return "*"; }
};

/// Division expression.
class DivExpr : public BinaryExpr {
public:
  using BinaryExpr::BinaryExpr;
  constexpr llvm::StringRef op() const override { return "/"; }
};

/// Negation expression.
class NegExpr : public UnaryExpr {
public:
  using UnaryExpr::UnaryExpr;
  constexpr llvm::StringRef op() const override { return "-"; }
};

/// Equality expression.
class EqExpr : public BinaryExpr {
public:
  using BinaryExpr::BinaryExpr;
  constexpr llvm::StringRef op() const override { return "="; }
};

/// Less-than expression.
class LeExpr : public BinaryExpr {
public:
  using BinaryExpr::BinaryExpr;
  constexpr llvm::StringRef op() const override { return "<"; }
};

/// Less or equal than expression.
class LeqExpr : public BinaryExpr {
public:
  using BinaryExpr::BinaryExpr;
  constexpr llvm::StringRef op() const override { return "<="; }
};

/// Greather-than expression.
class GeExpr : public BinaryExpr {
public:
  using BinaryExpr::BinaryExpr;
  constexpr llvm::StringRef op() const override { return ">"; }
};

/// Greather or equal than expression.
class GeqExpr : public BinaryExpr {
public:
  using BinaryExpr::BinaryExpr;
  constexpr llvm::StringRef op() const override { return ">="; }
};

} // namespace picus

namespace llvm {
raw_ostream &operator<<(raw_ostream &, const picus::Expression &);
raw_ostream &operator<<(raw_ostream &, const picus::Expression::ptr &);
} // namespace llvm
