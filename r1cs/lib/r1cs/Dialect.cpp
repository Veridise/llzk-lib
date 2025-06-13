//===-- Ops.cpp - R1CS dialect implementation ---------------*- C++ -*-----===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "r1cs/Dialect/IR/Dialect.h"

#include "mlir/IR/DialectImplementation.h"

// These two lines are critical:
#include "r1cs/Dialect/IR/Ops.h"
#include "r1cs/Dialect/IR/Types.h"

// TableGen'd implementation files
#include "r1cs/Dialect/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "r1cs/Dialect/IR/Types.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "r1cs/Dialect/IR/Attrs.cpp.inc"

using namespace mlir;
using namespace r1cs;

auto R1CSDialect::initialize() -> void {
  addOperations<
#define GET_OP_LIST
#include "r1cs/Dialect/IR/Ops.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "r1cs/Dialect/IR/Types.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "r1cs/Dialect/IR/Attrs.cpp.inc"
      >();
}
mlir::Attribute R1CSDialect::parseAttribute(DialectAsmParser &parser, mlir::Type type) const {
  llvm::SMLoc loc = parser.getCurrentLocation();

  // Parse the mnemonic like `#r1cs.felt`
  StringRef attrMnemonic;
  if (failed(parser.parseKeyword(&attrMnemonic))) {
    return {};
  }

  if (attrMnemonic == "felt") {
    // Parse the `<5>` part
    if (failed(parser.parseLess())) {
      return {};
    }

    // Expect an integer
    llvm::APInt value;
    if (failed(parser.parseInteger(value))) {
      return {};
    }

    if (failed(parser.parseGreater())) {
      llvm::outs() << value << "\n";
      return {};
    }

    auto intAttr = parser.getBuilder().getIntegerAttr(
        parser.getBuilder().getIntegerType(value.getBitWidth(), /*isSigned=*/true), value
    );

    return FeltAttr::get(parser.getContext(), intAttr);
  }

  parser.emitError(loc, "unknown attribute mnemonic '") << attrMnemonic << "'";
  return {};
}

void R1CSDialect::printAttribute(Attribute attr, DialectAsmPrinter &printer) const {
  if (auto feltAttr = attr.dyn_cast<FeltAttr>()) {
    printer << "felt<" << feltAttr.getValue().getValue() << ">";
    return;
  }
  llvm_unreachable("Unknown r1cs attribute");
}
