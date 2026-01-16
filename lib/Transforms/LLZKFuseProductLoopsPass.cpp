//===-- LLZKComputeConstrainToProductPass.cpp -------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-fuse-product-loops` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <memory>
namespace llzk {
#define GEN_PASS_DECL_FUSEPRODUCTLOOPSPASS
#define GEN_PASS_DEF_FUSEPRODUCTLOOPSPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

namespace llzk {
std::unique_ptr<mlir::Pass> createFuseProductLoopsPass() { return nullptr; }
} // namespace llzk
