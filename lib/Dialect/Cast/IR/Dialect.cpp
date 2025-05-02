//===-- Dialect.cpp - Cast dialect implementation ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Cast/IR/Ops.h"

// TableGen'd implementation files
#include "llzk/Dialect/Cast/IR/Dialect.cpp.inc"

//===------------------------------------------------------------------===//
// CastDialect
//===------------------------------------------------------------------===//

auto llzk::cast::CastDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Cast/IR/Ops.cpp.inc"
  >();
  // clang-format on
}
