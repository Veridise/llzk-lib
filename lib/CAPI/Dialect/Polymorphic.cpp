//===-- Polymorphic.cpp - Polymorphic dialect C API impl --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <llzk/Dialect/Polymorphic/IR/Dialect.h>
#include <llzk/Dialect/Polymorphic/IR/Types.h>
#include <llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h>

#include <mlir/CAPI/Pass.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>

#include <llzk-c/Dialect/Bool.h>
#include <mlir-c/Pass.h>

using namespace llzk::polymorphic;

// Include impl for transformation passes
#include <llzk/Dialect/Polymorphic/Transforms/TransformationPasses.capi.cpp.inc>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(
    Polymorphic, llzk__polymorphic, llzk::polymorphic::PolymorphicDialect
)
