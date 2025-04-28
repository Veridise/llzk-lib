//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_ARRAY_DIALECT
#define LLZK_ARRAY_DIALECT

include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/DialectBase.td"

def ArrayDialect : Dialect {
  let name = "array";
  let summary = "LLZK array operations.";

  let cppNamespace = "::llzk::array";

  let dependentDialects = ["::mlir::BuiltinDialect"];
}

#endif // LLZK_ARRAY_DIALECT
