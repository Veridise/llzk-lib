//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_OPS
#define LLZK_OPS

include "llzk/Dialect/LLZK/IR/Attrs.td"
include "llzk/Dialect/LLZK/IR/OpsBase.td"
include "llzk/Dialect/Array/IR/Types.td"
include "llzk/Dialect/Function/IR/OpTraits.td"
include "llzk/Dialect/Felt/IR/Types.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

// Note: `resultType.builderCall` must not be empty
class LLZK_BinaryOpBase<string mnemonic, Type resultType,
                        list<Trait> traits = []>
    : BinaryOpBase<LLZKDialect, mnemonic, resultType, traits>;

// Note: `resultType.builderCall` must not be empty
class LLZK_UnaryOpBase<string mnemonic, Type resultType,
                       list<Trait> traits = []>
    : UnaryOpBase<LLZKDialect, mnemonic, resultType, traits>;

class LLZK_Op<string mnemonic, list<Trait> traits = []>
    : Op<LLZKDialect, mnemonic, traits>;


#endif // LLZK_OPS
