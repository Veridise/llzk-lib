//===-- Types.cpp - LLZK type implementations -------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/Util/ArrayTypeHelper.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Types.h"
#include "llzk/Util/AttributeHelper.h"
#include "llzk/Util/ErrorHelper.h"
#include "llzk/Util/StreamHelper.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

#include <cassert>

namespace llzk {

using namespace mlir;

//===------------------------------------------------------------------===//
// StructType
//===------------------------------------------------------------------===//

LogicalResult StructType::verify(EmitErrorFn emitError, SymbolRefAttr nameRef, ArrayAttr params) {
  return verifyStructTypeParams(emitError, params);
}

FailureOr<SymbolLookupResult<StructDefOp>>
StructType::getDefinition(SymbolTableCollection &symbolTable, Operation *op) const {
  // First ensure this StructType passes verification
  ArrayAttr typeParams = this->getParams();
  if (failed(StructType::verify([op] { return op->emitError(); }, getNameRef(), typeParams))) {
    return failure();
  }
  // Perform lookup and ensure the symbol references a StructDefOp
  auto res = lookupTopLevelSymbol<StructDefOp>(symbolTable, getNameRef(), op);
  if (failed(res) || !res.value()) {
    return op->emitError() << "could not find '" << StructDefOp::getOperationName() << "' named \""
                           << getNameRef() << "\"";
  }
  // If this StructType contains parameters, make sure they match the number from the StructDefOp.
  if (typeParams) {
    auto defParams = res.value().get().getConstParams();
    size_t numExpected = defParams ? defParams->size() : 0;
    if (typeParams.size() != numExpected) {
      return op->emitError() << "'" << StructType::name << "' type has " << typeParams.size()
                             << " parameters but \"" << res.value().get().getSymName()
                             << "\" expects " << numExpected;
    }
  }
  return res;
}

LogicalResult StructType::verifySymbolRef(SymbolTableCollection &symbolTable, Operation *op) {
  return getDefinition(symbolTable, op);
}

LogicalResult StructType::hasColumns(SymbolTableCollection &symbolTable, Operation *op) const {
  auto lookup = getDefinition(symbolTable, op);
  if (failed(lookup)) {
    return lookup;
  }
  return lookup->get().hasColumns();
}

} // namespace llzk
