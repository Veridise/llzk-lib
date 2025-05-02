//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_STRUCT_DIALECT
#define LLZK_STRUCT_DIALECT

include "mlir/IR/DialectBase.td"

def StructDialect : Dialect {
  let name = "struct";
  let summary = "LLZK component/struct operations.";

  let cppNamespace = "::llzk::component";

  let useDefaultTypePrinterParser = true;
}

#endif // LLZK_STRUCT_DIALECT
