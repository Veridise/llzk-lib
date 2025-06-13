//===-- Ops.cpp - R1CS operation implementations ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "r1cs/Dialect/IR/Ops.h"

using namespace mlir;
using namespace r1cs;

#define GET_OP_CLASSES
#include "r1cs/Dialect/IR/Ops.cpp.inc"

mlir::Block *CircuitOp::addEntryBlock() {
  Region &body = getBody();
  assert(body.empty() && "CircuitOp already has a block");
  Block *block = new Block();
  body.push_back(block);
  return block;
}

mlir::ParseResult CircuitOp::parse(OpAsmParser &parser, OperationState &state) {
  // Parse the circuit name (symbol name).
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), state.attributes)) {
    return failure();
  }

  // Parse `inputs` keyword and argument list with types.
  if (parser.parseKeyword("inputs") || parser.parseLParen()) {
    return failure();
  }

  SmallVector<OpAsmParser::Argument> inputArgs;
  if (parser.parseArgumentList(inputArgs, AsmParser::Delimiter::None, /*allowType=*/true) ||
      parser.parseRParen()) {
    return failure();
  }

  // Just a marker keyword — outputs are defined inside the region via `r1cs.return`.
  // We ignore the actual type list here since CircuitOp has no `outs`.

  // Parse region with block arguments.
  Region *body = state.addRegion();
  if (parser.parseRegion(*body, inputArgs)) {
    return failure();
  }

  // Parse optional attributes.
  if (parser.parseOptionalAttrDictWithKeyword(state.attributes)) {
    return failure();
  }

  return success();
}

void CircuitOp::print(OpAsmPrinter &p) {
  // Print the circuit symbol name.
  p << ' ' << getSymName();

  // Print inputs.
  p << " inputs (";
  llvm::interleaveComma(getBody().front().getArguments(), p, [&](mlir::Value arg) {
    p << arg << " : " << arg.getType();
  });
  p << ")";

  // Print region (print block args = false, since we already printed above).
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/true);

  // Print attributes, excluding sym_name.
  p.printOptionalAttrDictWithKeyword(
      getOperation()->getAttrs(), {mlir::SymbolTable::getSymbolAttrName()}
  );
}
