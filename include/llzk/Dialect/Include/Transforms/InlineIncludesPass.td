//===-- InlineIncludesPass.td ------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_INCLUDE_INLINE_INCLUDES_PASS_TD
#define LLZK_INCLUDE_INLINE_INCLUDES_PASS_TD

include "mlir/Pass/PassBase.td"

def InlineIncludesPass : Pass<"llzk-inline-includes", "::mlir::ModuleOp"> {
  let summary = "Replace all IncludeOp with contents of the referenced file.";
  let constructor = "llzk::include::createInlineIncludesPass()";

  // LLZKDialect is dependent as encountering an LLZK op during the inlining
  // may cause a crash otherwise, as dialects are loaded lazily.
  let dependentDialects = ["llzk::include::IncludeDialect",
                           "llzk::LLZKDialect"];
}

#endif // LLZK_INCLUDE_INLINE_INCLUDES_PASS_TD
