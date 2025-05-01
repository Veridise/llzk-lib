//===-- Ops.cpp - Boolean operation implementations ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Types.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Bool/IR/Ops.cpp.inc"

using namespace mlir;

namespace llzk::boolean {

//===------------------------------------------------------------------===//
// AssertOp
//===------------------------------------------------------------------===//

// This side effect models "program termination".
// Based on
// https://github.com/llvm/llvm-project/blob/f325e4b2d836d6e65a4d0cf3efc6b0996ccf3765/mlir/lib/Dialect/ControlFlow/IR/ControlFlowOps.cpp#L92-L97
void AssertOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects
) {
  effects.emplace_back(MemoryEffects::Write::get());
}

} // namespace llzk::boolean
