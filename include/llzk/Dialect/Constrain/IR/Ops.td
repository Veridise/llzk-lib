//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_CONSTRAIN_OPS
#define LLZK_CONSTRAIN_OPS

include "llzk/Dialect/Constrain/IR/Dialect.td"
include "llzk/Dialect/Constrain/IR/OpInterfaces.td"
include "llzk/Dialect/Function/IR/OpTraits.td"
include "llzk/Dialect/Shared/Ops.td"
include "llzk/Dialect/LLZK/IR/Types.td"
include "llzk/Dialect/Array/IR/Types.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===------------------------------------------------------------------===//
// Constraint emit operations
//===------------------------------------------------------------------===//

class EmitOp<string mnemonic, list<Trait> traits = []>
    : Op<ConstrainDialect, mnemonic,
              traits#[ConstraintOpInterface, ConstraintGen,
                      DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {

  let assemblyFormat = [{
    $lhs `,` $rhs `:` type($lhs)
    `` custom<InferredOrParsedType>(type($rhs), ref(type($lhs)))
    attr-dict
  }];

  let extraClassDeclaration = [{
    static ::mlir::Type inferRHS(::mlir::Type lhsType);

    static ::mlir::ParseResult parseInferredOrParsedType(::mlir::OpAsmParser &parser,
        ::mlir::Type &rhsType, ::mlir::Type lhsType
    ) {
      if (mlir::succeeded(parser.parseOptionalComma())) {
        // If there is a comma, parse the `rhsType`
        mlir::Type type;
        if (parser.parseCustomTypeWithFallback(type)) {
          return mlir::failure();
        }
        rhsType = type;
      } else {
        // Otherwise, infer the `rhsType` from `lhsType`
        rhsType = inferRHS(lhsType);
      }
      return mlir::success();
    }

    static void printInferredOrParsedType(::mlir::OpAsmPrinter &printer,
        ::mlir::Operation *op, ::mlir::Type rhsType, ::mlir::Type lhsType
    ) {
      printer << ", ";
      printer.printStrippedAttrOrType(rhsType);
    }
  }];
}

def EmitEqualityOp
    : EmitOp<"eq", [Commutative, ElementwiseMappable,
                              TypesUnify<"lhs", "rhs">,
]> {
  let summary = "emit an equality constraint";
  let description = [{
    Emits an equality constraint between lhs and rhs. For the built-in Signal
    struct, the constraint is performed on the Signal's inner `@reg` felt field.
  }];

  let arguments = (ins EmitEqType:$lhs, EmitEqType:$rhs);
}

def EmitContainmentOp
    : EmitOp<"in", [ArrayElemTypeUnifyWith<"lhs", "rhs">]> {
  let summary = "emit a containment constraint";
  let description = [{}];

  let arguments = (ins LLZK_ArrayType:$lhs, ArrayElemType:$rhs);
}

#endif // LLZK_CONSTRAIN_OPS
