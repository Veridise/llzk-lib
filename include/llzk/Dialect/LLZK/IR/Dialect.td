//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_DIALECT
#define LLZK_DIALECT

include "mlir/IR/DialectBase.td"

def LLZKDialect : Dialect {
  let name = "llzk";
  let summary =
      "Veridise's intermediate representation for zero knowledge languages.";
  let description = [{
    This dialect contains types and operations for zero knowledge languages.
  }];

  let cppNamespace = "llzk";

  let dependentDialects = ["::mlir::scf::SCFDialect"];

  let useDefaultTypePrinterParser = true;
  let useDefaultAttributePrinterParser = true;
}

#endif // LLZK_DIALECT
