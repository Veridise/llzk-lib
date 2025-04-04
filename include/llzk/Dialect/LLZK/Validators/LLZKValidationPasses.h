//===-- LLZKValidationPasses.h - Validation Passes --------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Pass/Pass.h>

namespace llzk {

std::unique_ptr<mlir::Pass> createFieldWriteValidatorPass();

#define GEN_PASS_REGISTRATION
#include "llzk/Dialect/LLZK/Validators/LLZKValidationPasses.h.inc"

}; // namespace llzk
