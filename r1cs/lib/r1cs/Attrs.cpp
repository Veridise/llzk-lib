//===-- Attrs.cpp - R1CS attribute implementations ---------------*- C++ -*===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "r1cs/Dialect/IR/Attrs.h"
#include "r1cs/Dialect/IR/Dialect.h"

#include <mlir/IR/AttrTypeSubElements.h>
#include <mlir/IR/Builders.h>           // for mlir::Builder
#include <mlir/Support/LLVM.h>          // for LLVM utilities
#include <mlir/Support/LogicalResult.h> // for LogicalResult
#include <mlir/Support/TypeID.h>        // for type IDs

#include <llvm/ADT/TypeSwitch.h> // for llvm::TypeSwitch

#define GET_ATTRDEF_CLASSES
#include "r1cs/Dialect/IR/Attrs.cpp.inc"
