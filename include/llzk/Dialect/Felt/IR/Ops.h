//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/OpTraits.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/Felt/IR/Ops.h.inc"
