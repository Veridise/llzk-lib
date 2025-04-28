//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Array/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Types.h"
#include "llzk/Util/OpHelper.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/Array/IR/OpInterfaces.h.inc"

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/Array/IR/Ops.h.inc"
