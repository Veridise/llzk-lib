//===-- Types.cpp - R1CS type implementations ---------------*- C++ -*-----===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "r1cs/Dialect/IR/Types.h"

#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace r1cs;

// For now, no extra logic — just compile the generated types.
#define GET_TYPEDEF_CLASSES
#include "r1cs/Dialect/IR/Types.cpp.inc"
