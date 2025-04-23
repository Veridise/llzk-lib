//===-- Statement.cpp - Picus statements implementations --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llzk/Target/Picus/Language/Expression.h>
#include <llzk/Target/Picus/Language/Statement.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

namespace picus {

//===----------------------------------------------------------------------===//
// InputStmt
//===----------------------------------------------------------------------===//

InputStmt::InputStmt(VarExpr Var) : var(Var) {}

void InputStmt::print(llvm::raw_ostream &os) const { os << "(input " << var << ")"; }

//===----------------------------------------------------------------------===//
// OutputStmt
//===----------------------------------------------------------------------===//

OutputStmt::OutputStmt(VarExpr Var) : var(Var) {}

void OutputStmt::print(llvm::raw_ostream &os) const { os << "(output " << var << ")"; }

//===----------------------------------------------------------------------===//
// AssertStmt
//===----------------------------------------------------------------------===//

AssertStmt::AssertStmt(Expression::ptr Expr) : expr(std::move(Expr)) { assert(expr); }

void AssertStmt::print(llvm::raw_ostream &os) const { os << "(assert " << *expr << ")"; }

//===----------------------------------------------------------------------===//
// LookupStmt
//===----------------------------------------------------------------------===//

LookupStmt::LookupStmt(Expression::ptr Expr, Expression::ptr Fixed) {
  exprs.push_back(std::move(Expr));
  fixedRefs.push_back(std::move(Fixed));
}

LookupStmt::LookupStmt(
    llvm::MutableArrayRef<Expression::ptr> Exprs, llvm::MutableArrayRef<Expression::ptr> FixedRefs
) {
  assert(Exprs.size() == FixedRefs.size() && !Exprs.empty());
  for (Expression::ptr &expr : Exprs) {
    exprs.push_back(std::move(expr));
  }
  for (Expression::ptr &fixed : FixedRefs) {
    fixedRefs.push_back(std::move(fixed));
  }
}

void LookupStmt::print(llvm::raw_ostream &os) const {
  os << "(lookup [";
  llvm::interleave(exprs, os, " ");
  os << "] [";
  llvm::interleave(fixedRefs, os, " ");
  os << "])";
}

//===----------------------------------------------------------------------===//
// CallStmt
//===----------------------------------------------------------------------===//

CallStmt::CallStmt(CallStmtData data)
    : moduleName(data.moduleName), inputs(data.inputs), outputs(data.outputs) {}

void CallStmt::print(llvm::raw_ostream &os) const {
  os << "(call [";
  llvm::interleave(outputs, os, " ");
  os << "] " << moduleName << "[";
  llvm::interleave(inputs, os, " ");
  os << "])";
}

} // namespace picus

llvm::raw_ostream &llvm::operator<<(llvm::raw_ostream &os, const picus::Statement &stmt) {
  stmt.print(os);
  return os;
}

llvm::raw_ostream &
llvm::operator<<(llvm::raw_ostream &os, const std::unique_ptr<picus::Statement> &stmt) {
  assert(stmt);
  stmt->print(os);
  return os;
}

llvm::raw_ostream &llvm::operator<<(llvm::raw_ostream &os, const picus::InputStmt &stmt) {
  stmt.print(os);
  return os;
}

llvm::raw_ostream &llvm::operator<<(llvm::raw_ostream &os, const picus::OutputStmt &stmt) {
  stmt.print(os);
  return os;
}

llvm::raw_ostream &llvm::operator<<(llvm::raw_ostream &os, const picus::AssertStmt &stmt) {
  stmt.print(os);
  return os;
}
