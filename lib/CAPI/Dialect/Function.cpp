//===-- Function.cpp - Function dialect C API implementation ----*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llzk/Dialect/Function/IR/Dialect.h>

#include <mlir/CAPI/Pass.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>

#include <mlir-c/Pass.h>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Function, llzk__function, llzk::function::FunctionDialect)
