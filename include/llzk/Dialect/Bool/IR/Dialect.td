//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_BOOLEAN_DIALECT
#define LLZK_BOOLEAN_DIALECT

include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/DialectBase.td"

def BoolDialect : Dialect {
  let name = "bool";
  let summary = "LLZK boolean operations.";

  let cppNamespace = "::llzk::boolean";

  let dependentDialects = ["::mlir::BuiltinDialect", "::llzk::felt::FeltDialect", "::llzk::function::FunctionDialect"];

  let useDefaultAttributePrinterParser = true;
}

#endif // LLZK_BOOLEAN_DIALECT
