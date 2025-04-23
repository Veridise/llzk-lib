//===-- Circuit.cpp - Picus program implementations -------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llzk/Dialect/LLZK/Util/AttributeHelper.h>
#include <llzk/Target/Picus/Language/Circuit.h>
#include <llzk/Target/Picus/Language/Statement.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

namespace picus {

//===----------------------------------------------------------------------===//
// Circuit
//===----------------------------------------------------------------------===//

Circuit::Circuit(PrimeNumber Prime) : prime(Prime) {}

void Circuit::print(llvm::raw_ostream &os) const {
  prime.print(os);
  for (auto &mod : modules) {
    mod.getValue().print(os);
  }
  fixed.print(os);
}

Module &Circuit::emplaceModule(llvm::StringRef name) {
  if (!modules.contains(name)) {
    modules.insert({name, Module(name)});
  }
  return (*modules.find(name)).getValue();
}

void FixedValues::addFixedValues(llvm::StringRef name, ConstExpr expr) {
  // Insert single ConstExpr into values[name]
  values[name].push_back(expr);
}

void FixedValues::addFixedValues(llvm::StringRef name, llvm::ArrayRef<ConstExpr> exprs) {
  // Append array of ConstExpr to values[name]
  for (const auto &e : exprs) {
    values[name].push_back(e);
  }
}

Expression::ptr FixedValues::getFixedValueRef(llvm::StringRef name) {
  // Dummy: Return null or wrap first expression
  auto it = values.find(name);
  if (it == values.end() || it->second.empty()) {
    llvm::errs() << "No fixed value found for name: " << name << "\n";
    return nullptr;
  }

  // Wrap ConstExpr into Expression::ptr if needed
  return std::make_unique<ConstExpr>(llzk::toAPInt(std::distance(values.begin(), it)));
}

void FixedValues::print(llvm::raw_ostream &os) const {
  os << "(fixed \n";
  for (const auto &entry : values) {
    os << "[";
    llvm::interleave(entry.second, os, " ");
    os << "]\n";
  }
  os << ")\n";
}

//===----------------------------------------------------------------------===//
// Module
//===----------------------------------------------------------------------===//

Module::Module(llvm::StringRef Name) : name(Name) {}

void Module::print(llvm::raw_ostream &os) const {
  os << "(begin-module " << name << ")\n";
  llvm::interleave(statements, os, "\n");
  os << "\n(end-module)\n";
}

//===----------------------------------------------------------------------===//
// PrimeNumber
//===----------------------------------------------------------------------===//

PrimeNumber::PrimeNumber(llvm::APInt Prime) : prime(Prime) {}

void PrimeNumber::print(llvm::raw_ostream &os) const { os << "(prime-number " << prime << ")"; }

} // namespace picus
