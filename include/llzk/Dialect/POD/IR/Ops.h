//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/POD/IR/Dialect.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Shared/OpHelpers.h"

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/POD/IR/Ops.h.inc"
