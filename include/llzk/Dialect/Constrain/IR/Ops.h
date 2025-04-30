//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/OpInterfaces.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/OpTraits.h"
#include "llzk/Dialect/LLZK/IR/Types.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/Constrain/IR/Ops.h.inc"
