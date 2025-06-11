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

mlir::ParseResult CircuitOp::parse(OpAsmParser &parser, OperationState &state) {
  // Parse the circuit name.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), state.attributes)) {
    return failure();
  }

  // Parse `inputs` keyword and input arguments with types.
  if (parser.parseKeyword("inputs")) {
    return failure();
  }

  SmallVector<OpAsmParser::Argument> inputArgs;
  SmallVector<Type> inputTypes;
  if (parser.parseArgumentList(inputArgs, AsmParser::Delimiter::None, /*allowType=*/true)) {
    return failure();
  }

  for (auto &arg : inputArgs) {
    inputTypes.push_back(arg.type);
  }

  // Parse `outputs` keyword and result types.
  if (parser.parseKeyword("outputs") || parser.parseColon()) {
    return failure();
  }

  SmallVector<Type> resultTypes;
  if (parser.parseTypeList(resultTypes)) {
    return failure();
  }
  state.addTypes(resultTypes); // Add to result types of op

  // Parse the region
  Region *region = state.addRegion();
  if (parser.parseRegion(*region, inputArgs, /*argTypes=*/ {})) {
    return failure();
  }

  // Parse optional attributes
  if (parser.parseOptionalAttrDictWithKeyword(state.attributes)) {
    return failure();
  }

  return success();
}

void CircuitOp::print(OpAsmPrinter &p) {
  p << ' ' << getSymName(); // Symbol name

  // Inputs
  p << "\n  inputs ";
  auto args = getBody().front().getArguments();
  llvm::interleaveComma(args, p, [&](mlir::Value arg) { p << arg << " : " << arg.getType(); });

  // Outputs
  p << "\n  outputs : ";
  llvm::interleaveComma(getResultTypes(), p);

  // Region
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);

  // Attributes
  p.printOptionalAttrDictWithKeyword(
      getOperation()->getAttrs(), {SymbolTable::getSymbolAttrName()}
  );
}
