//===-- Constrain.cpp - Constrain dialect C API implementation --*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llzk/Dialect/Constrain/IR/Dialect.h>

#include <mlir/CAPI/Registration.h>

#include <llzk-c/Dialect/Constrain.h>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Constrain, llzk__constrain, llzk::constrain::ConstrainDialect)
