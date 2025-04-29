//===-- InlineIncludesPass.h ------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Pass/PassBase.h"

namespace llzk::include {

std::unique_ptr<mlir::Pass> createInlineIncludesPass();

#define GEN_PASS_REGISTRATION
#include "llzk/Dialect/Include/Transforms/InlineIncludesPass.h.inc"

}; // namespace llzk::include
