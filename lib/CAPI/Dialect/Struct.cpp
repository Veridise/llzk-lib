//===-- Struct.cpp - Struct dialect C API implementation --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llzk/Dialect/Struct/IR/Dialect.h>
#include <llzk/Dialect/Struct/IR/Types.h>

#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Struct, llzk__component, llzk::component::StructDialect)
