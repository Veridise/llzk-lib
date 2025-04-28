//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_FUNCTION_DIALECT
#define LLZK_FUNCTION_DIALECT

include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/DialectBase.td"

def FunctionDialect : Dialect {
  let name = "function";
  let summary = "LLZK function operations.";

  let cppNamespace = "::llzk::function";

  let dependentDialects = ["::mlir::BuiltinDialect"];
}

#endif // LLZK_FUNCTION_DIALECT
