//===-- Ops.cpp - R1CS operation implementations ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "r1cs/Dialect/IR/Ops.h"

#include <mlir/IR/OpImplementation.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "r1cs/Dialect/IR/Ops.cpp.inc"

using namespace mlir;
namespace r1cs {

ParseResult CircuitDefOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr symName;
  if (parser.parseSymbolName(symName, "sym_name", result.attributes)) {
    return failure();
  }

  // Optional inputs (...)
  SmallVector<OpAsmParser::Argument> args;
  Type argType;
  if (succeeded(parser.parseOptionalKeyword("inputs"))) {
    if (parser.parseLParen()) {
      return failure();
    }

    do {
      OpAsmParser::Argument arg;
      if (parser.parseArgument(arg) || parser.parseColonType(argType)) {
        return failure();
      }
      arg.type = argType;
      args.push_back(arg);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRParen()) {
      return failure();
    }
  }

  // Parse optional attribute dict
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  // Parse body region with those args
  Region *body = result.addRegion();
  return parser.parseRegion(*body, args);
}

void CircuitDefOp::print(mlir::OpAsmPrinter &p) {
  p << " ";
  p.printSymbolName(getSymName());

  Block &entry = getBody().front();
  if (!entry.empty()) {
    p << " inputs (";
    llvm::interleaveComma(entry.getArguments(), p, [&](BlockArgument arg) {
      p << arg << ": ";
      if (auto sigTy = arg.getType().dyn_cast<SignalType>()) {
        sigTy.print(p);
      } else {
        p.printType(arg.getType()); // fallback for robustness
      }
    });
    p << ")";
  }

  p.printOptionalAttrDict((*this)->getAttrs(), {"sym_name"});
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

mlir::Block *CircuitDefOp::addEntryBlock() {
  Region &body = getBody();
  assert(body.empty() && "CircuitOp already has a block");
  Block *block = new Block();
  body.push_back(block);
  return block;
}

void CircuitDefOp::build(OpBuilder &builder, OperationState &state, llvm::StringRef name) {
  state.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
  state.addRegion();
}
} // namespace r1cs
