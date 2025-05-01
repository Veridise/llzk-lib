//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Function/IR/OpTraits.h"
#include "llzk/Dialect/Global/IR/Dialect.h"
#include "llzk/Util/SymbolLookup.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpImplementation.h>

// forward-declare ops
#define GET_OP_FWD_DEFINES
#include "llzk/Dialect/Global/IR/Ops.h.inc"

// Include TableGen'd declarations
#include "llzk/Dialect/Global/IR/OpInterfaces.h.inc"

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/Global/IR/Ops.h.inc"
