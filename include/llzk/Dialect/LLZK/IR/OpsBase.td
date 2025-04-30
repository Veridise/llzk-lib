//===-- OpsBase.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_OPS_BASE
#define LLZK_OPS_BASE

include "llzk/Dialect/LLZK/IR/Types.td"
include "llzk/Dialect/LLZK/IR/OpInterfaces.td"
include "llzk/Dialect/Shared/Ops.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Base Op classes
//===----------------------------------------------------------------------===//

// Note: `resultType.builderCall` must not be empty
class BinaryOpBase<Dialect dialect, string mnemonic, Type resultType,
                   list<Trait> traits = []>
    : NaryOpBase<dialect, mnemonic, resultType.builderCall,
                 traits#[Pure, TypeUnifyWithResult<"lhs">,
                         TypeUnifyWithResult<"rhs">]> {

  let arguments = (ins VarTypeOr<resultType>:$lhs, VarTypeOr<resultType>:$rhs);
  let results = (outs resultType:$result);

  let assemblyFormat = [{
      $lhs `,` $rhs
      `` custom<InferredOrParsedType>(type($lhs), "true")
      `` custom<InferredOrParsedType>(type($rhs), "false")
      attr-dict
  }];
}

// Note: `resultType.builderCall` must not be empty
class UnaryOpBase<Dialect dialect, string mnemonic, Type resultType,
                  list<Trait> traits = []>
    : NaryOpBase<dialect, mnemonic, resultType.builderCall,
                 traits#[Pure, TypeUnifyWithResult<"operand">]> {

  let arguments = (ins VarTypeOr<resultType>:$operand);
  let results = (outs resultType:$result);

  let assemblyFormat = [{
      $operand
      `` custom<InferredOrParsedType>(type($operand), "true")
      attr-dict
  }];
}

class LLZK_Op<string mnemonic, list<Trait> traits = []>
    : Op<LLZKDialect, mnemonic, traits>;

class LLZK_GlobalRefOp<string mnemonic, list<Trait> traits = []>
    : LLZK_Op<mnemonic,
              traits#[DeclareOpInterfaceMethods<GlobalRefOpInterface>,
                      DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let extraClassDeclaration = [{
    /// Gets the definition for the `global` referenced in this op.
    inline ::mlir::FailureOr<SymbolLookupResult<GlobalDefOp>> getGlobalDefOp(::mlir::SymbolTableCollection &tables) {
      return ::llvm::cast<GlobalRefOpInterface>(getOperation()).getGlobalDefOp(tables);
    }
  }];
}

// Note: `resultType.builderCall` must not be empty
class LLZK_BinaryOpBase<string mnemonic, Type resultType,
                        list<Trait> traits = []>
    : BinaryOpBase<LLZKDialect, mnemonic, resultType, traits>;

// Note: `resultType.builderCall` must not be empty
class LLZK_UnaryOpBase<string mnemonic, Type resultType,
                       list<Trait> traits = []>
    : UnaryOpBase<LLZKDialect, mnemonic, resultType, traits>;

#endif // LLZK_OPS_BASE
