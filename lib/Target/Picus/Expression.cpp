//===-- Expression.cpp - Picus expressions implementations ------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llzk/Target/Picus/Language/Expression.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

namespace picus {

//===----------------------------------------------------------------------===//
// VarExpr
//===----------------------------------------------------------------------===//

VarExpr::VarExpr(llvm::StringRef Name) : name(Name) {}

void VarExpr::print(llvm::raw_ostream &os) const { os << "x" << name; }

//===----------------------------------------------------------------------===//
// ConstExpr
//===----------------------------------------------------------------------===//

ConstExpr::ConstExpr(llvm::APInt Value) : value(Value) {}

void ConstExpr::print(llvm::raw_ostream &os) const { os << value; }

//===----------------------------------------------------------------------===//
// BinaryExpr
//===----------------------------------------------------------------------===//

BinaryExpr::BinaryExpr(Expression::ptr LHS, Expression::ptr RHS)
    : lhs(std::move(LHS)), rhs(std::move(RHS)) {
  assert(lhs);
  assert(rhs);
}

void BinaryExpr::print(llvm::raw_ostream &os) const {
  os << "(" << op() << " " << lhs << " " << rhs << ")";
}

//===----------------------------------------------------------------------===//
// UnaryExpr
//===----------------------------------------------------------------------===//

UnaryExpr::UnaryExpr(Expression::ptr Value) : value(std::move(Value)) { assert(value); }

void UnaryExpr::print(llvm::raw_ostream &os) const { os << "(" << op() << " " << value << ")"; }

} // namespace picus

llvm::raw_ostream &llvm::operator<<(llvm::raw_ostream &os, const picus::Expression &expr) {
  expr.print(os);
  return os;
}

llvm::raw_ostream &llvm::operator<<(llvm::raw_ostream &os, const picus::Expression::ptr &expr) {
  assert(expr);
  expr->print(os);
  return os;
}
