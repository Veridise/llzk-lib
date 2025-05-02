//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_GLOBAL_DIALECT
#define LLZK_GLOBAL_DIALECT

include "mlir/IR/DialectBase.td"

def GlobalDialect : Dialect {
  let name = "global";
  let summary = "LLZK support for global values, defined outside of structs.";

  let cppNamespace = "::llzk::global";
}

#endif // LLZK_GLOBAL_DIALECT
