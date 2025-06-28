//===-- LLZKInlineStructsPass.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-inline-structs` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Transforms/LLZKTransformationPasses.h"

// Include the generated base pass class definitions.
namespace llzk {
// the *DECL* macro is required when a pass has options to declare the option struct
#define GEN_PASS_DECL_INLINESTRUCTSPASS
#define GEN_PASS_DEF_INLINESTRUCTSPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

namespace {

class InlineStructsPass : public llzk::impl::InlineStructsPassBase<InlineStructsPass> {
public:
  void runOnOperation() override {
    // TODO
    assert(false && "TODO: not yet implemented");
  }
};

} // namespace

std::unique_ptr<mlir::Pass> llzk::createInlineStructsPass() {
  return std::make_unique<InlineStructsPass>();
};
