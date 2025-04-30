//===-- Types.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Struct/IR/Dialect.h"
#include "llzk/Util/ErrorHelper.h"
#include "llzk/Util/SymbolLookup.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/MemorySlotInterfaces.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>

#include <vector>

// forward-declare ops
#define GET_OP_FWD_DEFINES
#include "llzk/Dialect/Struct/IR/Ops.h.inc"

// Include TableGen'd declarations
#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/Struct/IR/Types.h.inc"
