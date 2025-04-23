//===-- LLZKValidationPasses.td - Validation Passes --------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the `-llzk-validate-field-writes` pass.
///
//===----------------------------------------------------------------------===//

#ifndef LLZK_VALIDATION_PASSES_TD
#define LLZK_VALIDATION_PASSES_TD

include "mlir/Pass/PassBase.td"

def FieldWriteValidatorPass
    : Pass<"llzk-validate-field-writes", "::llzk::StructDefOp"> {
  let summary =
      "Detect multiple and missing writes to the same field of a component.";
  let constructor = "llzk::createFieldWriteValidatorPass()";
  let dependentDialects = ["llzk::LLZKDialect"];
}

#endif // LLZK_VALIDATION_PASSES_TD
