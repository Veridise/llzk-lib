//===-- Include.cpp - Include dialect C API implementation ------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Include/IR/Dialect.h"
#include "llzk/Dialect/Include/IR/Ops.h"
#include "llzk/Dialect/Include/Transforms/InlineIncludesPass.h"

#include "llzk-c/Dialect/Include.h"

#include <mlir/CAPI/Pass.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>

#include <mlir-c/Pass.h>

static void registerLLZKIncludeTransformationPasses() {
  llzk::include::registerTransformationPasses();
}

using namespace llzk::include;

// Include impl for transformation passes
#include <llzk/Dialect/Include/Transforms/InlineIncludesPass.capi.cpp.inc>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Include, llzk__include, llzk::include::IncludeDialect)

MlirOperation llzkIncludeOpCreate(MlirLocation location, MlirStringRef name, MlirStringRef path) {
  return wrap(IncludeOp::create(unwrap(location), unwrap(name), unwrap(path)));
}
