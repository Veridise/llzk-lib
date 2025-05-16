//===-- Cast.cpp - Cast dialect C API implementation ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llzk/Dialect/Cast/IR/Dialect.h>

#include <mlir/CAPI/Registration.h>

#include <llzk-c/Dialect/Bool.h>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Cast, llzk__cast, llzk::cast::CastDialect)
