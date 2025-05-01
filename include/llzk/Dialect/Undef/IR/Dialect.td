//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_UNDEF_DIALECT
#define LLZK_UNDEF_DIALECT

include "mlir/IR/DialectBase.td"

def UndefDialect : Dialect {
  let name = "undef";
  let summary = "LLZK Undefined value support operations.";

  let cppNamespace = "::llzk::undef";

  let dependentDialects = ["::mlir::BuiltinDialect"];
}

#endif // LLZK_UNDEF_DIALECT
