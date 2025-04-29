//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/OpInterfaces.h"
#include "llzk/Dialect/LLZK/IR/Types.h"
#include "llzk/Util/BuilderHelper.h"
#include "llzk/Util/Constants.h"
#include "llzk/Util/OpHelper.h"
#include "llzk/Util/SymbolLookup.h"

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Dialect/Affine/IR/AffineValueMap.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/LLZK/IR/Ops.h.inc"

namespace llzk {
// TODO: these should probably move to `struct` dialect

mlir::InFlightDiagnostic
genCompareErr(StructDefOp &expected, mlir::Operation *origin, const char *aspect);

mlir::LogicalResult checkSelfType(
    mlir::SymbolTableCollection &symbolTable, StructDefOp &expectedStruct, mlir::Type actualType,
    mlir::Operation *origin, const char *aspect
);

} // namespace llzk
