//===-- Parsers.h -----------------------------------------------*- C++ -*-===//
//
// Command line parsers for LLZK transformation passes.
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/CommandLine.h>

// Custom command line parsers
namespace llvm {
namespace cl {

// Parser for APInt
template <> class parser<APInt> : public basic_parser<APInt> {
public:
  parser(Option &O) : basic_parser(O) {}

  bool parse(Option &O, StringRef, StringRef Arg, APInt &Val) {
    if (Arg.empty()) {
      return O.error("empty integer literal");
    }
    // Decimal-only: allocate a safe width then shrink.
    unsigned bits = std::max(1u, 4u * (unsigned)Arg.size());
    APInt tmp(bits, Arg, 10);
    unsigned active = tmp.getActiveBits();
    if (active == 0) {
      active = 1;
    }
    Val = tmp.zextOrTrunc(active);
    return false;
  }

  // Prints how the passed option differs from the default one specified in the pass
  // For example, if V = 17 and Default = 11 then it should print
  // [OptionName] 17 (bits=5) (default: 11 (bits=4))
  void printOptionDiff(
      const Option &O, const APInt &V, OptionValue<APInt> Default, size_t GlobalWidth
  ) const {
    std::string Cur = llvm::toString(V, 10, false);
    Cur += " (bits=" + std::to_string(V.getBitWidth()) + ")";

    std::string Def = "<unspecified>";
    if (Default.hasValue()) {
      const APInt &D = Default.getValue();
      Def = llvm::toString(D, 10, false);
      Def += " (bits=" + std::to_string(D.getBitWidth()) + ")";
    }

    printOptionName(O, GlobalWidth);
    llvm::outs() << Cur << " (default: " << Def << ")\n";
  }
};

} // namespace cl
} // namespace llvm
