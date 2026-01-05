//===-- Undef.cpp - Undef dialect C API implementation ----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/Undef/IR/Dialect.h"
#include "llzk/Dialect/Undef/IR/Ops.h"

#include "llzk-c/Dialect/Undef.h"

#include <mlir/CAPI/Registration.h>

using namespace llzk::undef;

// Include the generated CAPI
#include "llzk/Dialect/Undef/IR/Ops.capi.cpp.inc"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Undef, llzk__undef, UndefDialect)
