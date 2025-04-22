//===-- Circuit.cpp - Picus program implementations -------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llzk/Target/Picus/Language/Circuit.h>
#include <llzk/Target/Picus/Language/Statement.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/raw_ostream.h>

namespace picus {

//===----------------------------------------------------------------------===//
// Circuit
//===----------------------------------------------------------------------===//

void Circuit::print(llvm::raw_ostream &os) const {
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

//===----------------------------------------------------------------------===//
// Module
//===----------------------------------------------------------------------===//

void Module::print(llvm::raw_ostream &os) const {
  os << "(begin-module " << name << ")\n";
  llvm::interleave(statements, os, "\n");
  os << "(end-module)\n";
}

} // namespace picus
