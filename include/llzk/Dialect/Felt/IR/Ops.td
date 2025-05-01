//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_FELT_OPS
#define LLZK_FELT_OPS

include "llzk/Dialect/Felt/IR/Dialect.td"
include "llzk/Dialect/Felt/IR/Types.td"
include "llzk/Dialect/Felt/IR/Attrs.td"
include "llzk/Dialect/Function/IR/OpTraits.td"
include "llzk/Dialect/Shared/OpsBase.td"

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===------------------------------------------------------------------===//
// Op Classes
//===------------------------------------------------------------------===//

class FeltDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<FeltDialect, mnemonic, traits>;

class FeltDialectBinaryOp<string mnemonic, Type resultType,
                          list<Trait> traits = []>
    : BinaryOpBase<FeltDialect, mnemonic, resultType, traits>;

class FeltDialectUnaryOp<string mnemonic, Type resultType,
                         list<Trait> traits = []>
    : UnaryOpBase<FeltDialect, mnemonic, resultType, traits>;

//===------------------------------------------------------------------===//
// Constants
//===------------------------------------------------------------------===//

def LLZK_FeltConstantOp
    : FeltDialectOp<"const", [ConstantLike, Pure,
                              DeclareOpInterfaceMethods<
                                  OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "field element constant";
  let description = [{
    This operation produces a felt-typed SSA value holding an integer constant.

    Example:

    ```llzk
    %0 = llzk.const 42
    ```
  }];

  let arguments = (ins LLZK_FeltConstAttr:$value);
  let results = (outs LLZK_FeltType:$result);
  let assemblyFormat = [{ $value attr-dict }];
  let hasFolder = 1;
}

def LLZK_FeltNonDetOp
    : FeltDialectOp<"nondet", [ConstantLike, Pure,
                               DeclareOpInterfaceMethods<
                                   OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "uninitialized field element";
  let description = [{
    This operation produces a felt-typed SSA value without a specified value.
    This can be used in `constrain()` functions in place of expressions that
    cannot be included in constraints.

    Example:

    ```llzk
    %0 = llzk.nondet
    ```
  }];

  let results = (outs LLZK_FeltType:$result);
  let assemblyFormat = [{ attr-dict }];
}

//===------------------------------------------------------------------===//
// Operators
//===------------------------------------------------------------------===//

def LLZK_AddFeltOp : FeltDialectBinaryOp<"add", LLZK_FeltType, [Commutative]> {
  let summary = "addition operator for field elements";
  let description = [{}];
}

def LLZK_SubFeltOp : FeltDialectBinaryOp<"sub", LLZK_FeltType> {
  let summary = "subtraction operator for field elements";
  let description = [{}];
}

def LLZK_MulFeltOp : FeltDialectBinaryOp<"mul", LLZK_FeltType, [Commutative]> {
  let summary = "multiplication operator for field elements";
  let description = [{}];
}

def LLZK_DivFeltOp : FeltDialectBinaryOp<"div", LLZK_FeltType> {
  let summary = "division operator for field elements";
  let description = [{}];
}

def LLZK_ModFeltOp : FeltDialectBinaryOp<"mod", LLZK_FeltType> {
  let summary = "modulus/remainder operator for field elements";
  let description = [{}];
}

def LLZK_NegFeltOp : FeltDialectUnaryOp<"neg", LLZK_FeltType> {
  let summary = "negation operator for field elements";
  let description = [{}];
}

def LLZK_InvFeltOp : FeltDialectUnaryOp<"inv", LLZK_FeltType, [WitnessGen]> {
  let summary = "inverse operator for field elements";
  let description = [{}];
}

def LLZK_AndFeltOp
    : FeltDialectBinaryOp<"bit_and", LLZK_FeltType, [WitnessGen, Commutative]> {
  let summary = "bitwise AND operator for field elements";
  let description = [{}];
}

def LLZK_OrFeltOp
    : FeltDialectBinaryOp<"bit_or", LLZK_FeltType, [WitnessGen, Commutative]> {
  let summary = "bitwise OR operator for field elements";
  let description = [{}];
}

def LLZK_XorFeltOp
    : FeltDialectBinaryOp<"bit_xor", LLZK_FeltType, [WitnessGen, Commutative]> {
  let summary = "bitwise XOR operator for field elements";
  let description = [{}];
}

def LLZK_NotFeltOp
    : FeltDialectUnaryOp<"bit_not", LLZK_FeltType, [WitnessGen]> {
  let summary = "bit flip operator for field elements";
  let description = [{}];
}

def LLZK_ShlFeltOp : FeltDialectBinaryOp<"shl", LLZK_FeltType, [WitnessGen]> {
  let summary = "left shift operator for field elements";
  let description = [{}];
}

def LLZK_ShrFeltOp : FeltDialectBinaryOp<"shr", LLZK_FeltType, [WitnessGen]> {
  let summary = "right shift operator for field elements";
  let description = [{}];
}

#endif // LLZK_FELT_OPS
