//===-- Ops.cpp - POD operation implementations -----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/Builders.h>

#include <llvm/ADT/SmallString.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/POD/IR/Ops.cpp.inc"

using namespace mlir;

namespace llzk::pod {}
