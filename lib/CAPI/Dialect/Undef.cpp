//===-- Undef.cpp - Undef dialect C API implementation ----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llzk/Dialect/Undef/IR/Dialect.h>

#include <mlir/CAPI/Registration.h>

#include <llzk-c/Dialect/Undef.h>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Undef, llzk__undef, llzk::undef::UndefDialect)
