//===-- OpInterfaces.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

// Include TableGen'd declarations
#include "llzk/Dialect/Constrain/IR/OpInterfaces.h.inc"

namespace llzk::constrain {

inline bool containsConstraintOp(mlir::Operation *op) {
  return op->walk([](ConstraintOpInterface p) { return mlir::WalkResult::interrupt(); }
  ).wasInterrupted();
}

} // namespace llzk::constrain
