//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_STRING_DIALECT
#define LLZK_STRING_DIALECT

include "mlir/IR/DialectBase.td"

def StringDialect : Dialect {
  let name = "string";
  let summary = "LLZK string type and operations.";

  let cppNamespace = "::llzk::string";

  let useDefaultTypePrinterParser = true;
}

#endif // LLZK_STRING_DIALECT
