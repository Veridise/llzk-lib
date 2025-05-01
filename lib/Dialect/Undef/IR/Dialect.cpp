//===-- Dialect.cpp - Undefined value operation implementation --*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Undef/IR/Dialect.h"
#include "llzk/Dialect/Undef/IR/Ops.h"

#include <mlir/IR/BuiltinDialect.h>

// TableGen'd implementation files
#include "llzk/Dialect/Undef/IR/Dialect.cpp.inc"

//===------------------------------------------------------------------===//
// UndefDialect
//===------------------------------------------------------------------===//

auto llzk::undef::UndefDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Undef/IR/Ops.cpp.inc"
  >();
  // clang-format on
}
