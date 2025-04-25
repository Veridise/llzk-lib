//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_INCLUDE_DIALECT
#define LLZK_INCLUDE_DIALECT

include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/DialectBase.td"

def IncludeDialect : Dialect {
  let name = "llzk_include";
  let summary =
      "LLZK include operations.";

  let cppNamespace = "::llzk::include";

  let dependentDialects = ["::mlir::BuiltinDialect"];
}

#endif // LLZK_INCLUDE_DIALECT
