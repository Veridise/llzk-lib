//===-- Types.cpp - R1CS type implementations ---------------*- C++ -*-----===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "r1cs/Dialect/IR/Types.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/TypeSupport.h>

using namespace mlir;
using namespace r1cs;

void SignalType::print(mlir::AsmPrinter &printer) const {
  printer << "signal";
  if (getIsPublic()) {
    printer << " public";
  }
}

mlir::Type SignalType::parse(mlir::AsmParser &parser) {
  bool isPublic = false;
  if (succeeded(parser.parseOptionalKeyword("public"))) {
    isPublic = true;
  }
  return get(parser.getContext(), isPublic);
}
