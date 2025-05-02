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

include "mlir/IR/DialectBase.td"

def IncludeDialect : Dialect {
  let name = "include";
  let summary = "LLZK include operations.";

  let cppNamespace = "::llzk::include";
}

#endif // LLZK_INCLUDE_DIALECT
