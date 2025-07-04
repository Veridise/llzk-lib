//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_BOOLEAN_OPS
#define LLZK_BOOLEAN_OPS

include "llzk/Dialect/Bool/IR/Dialect.td"
include "llzk/Dialect/Bool/IR/Attrs.td"
include "llzk/Dialect/Felt/IR/Types.td"
include "llzk/Dialect/Function/IR/OpTraits.td"
include "llzk/Dialect/Shared/OpsBase.td"

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===------------------------------------------------------------------===//
// Op Classes
//===------------------------------------------------------------------===//

class BoolDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<BoolDialect, mnemonic, traits>;

class BoolBinaryOpBase<string mnemonic, Type resultType,
                       list<Trait> traits = []>
    : BinaryOpBase<BoolDialect, mnemonic, resultType, traits>;

class BoolUnaryOpBase<string mnemonic, Type resultType, list<Trait> traits = []>
    : UnaryOpBase<BoolDialect, mnemonic, resultType, traits>;

//===------------------------------------------------------------------===//
// Boolean operators
//===------------------------------------------------------------------===//

def LLZK_AndBoolOp : BoolBinaryOpBase<"and", I1, [WitnessGen, Commutative]> {
  let summary = "logical AND operator";
  let description = [{
    This operation computes the logical AND (i.e., conjunction) of two `i1` (i.e., boolean)
    values as an `i1` value. The result is `1` if the operation is true and `0` otherwise.
  }];
}

def LLZK_OrBoolOp : BoolBinaryOpBase<"or", I1, [WitnessGen, Commutative]> {
  let summary = "logical OR operator";
  let description = [{
    This operation computes the logical OR (i.e., disjunction) of two `i1` (i.e., boolean)
    values as an `i1` value. The result is `1` if the operation is true and `0` otherwise.
  }];
}

def LLZK_XorBoolOp : BoolBinaryOpBase<"xor", I1, [WitnessGen, Commutative]> {
  let summary = "logical XOR operator";
  let description = [{
    This operation computes the logical XOR (i.e., exclusive disjunction) of two `i1` (i.e., boolean)
    values as an `i1` value. The result is `1` if the operation is true and `0` otherwise.
  }];
}

def LLZK_NotBoolOp : BoolUnaryOpBase<"not", I1, [WitnessGen]> {
  let summary = "logical NOT operator";
  let description = [{
    This operation computes the logical NOT (i.e., negation) of an `i1` (i.e., boolean)
    value as an `i1` value. The result is `1` if the operation is true and `0` otherwise.
  }];
}

//===------------------------------------------------------------------===//
// Other operators
//===------------------------------------------------------------------===//

def LLZK_AssertOp
    : BoolDialectOp<
          "assert", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "assertion operation";
  let description = [{
    This operation asserts that a given boolean value is true. Assertions are checked
    statically when possible. If the condition evaluates to `true`, the assertion is
    removed. If `false`, an error is reported. Otherwise, the assertion is preserved.
    All assertions that appear in `constrain()` functions must evaluate statically
    (i.e., they cannot depend on inputs to the circuit) else an error is reported.

    Assertion without message:
    ```llzk
    %1 = bool.cmp lt(%a, %b)
    bool.assert %1
    ```

    Assertion with a message:
    ```llzk
    %1 = bool.cmp eq(%a, %b)
    bool.assert %1, "expected equal values"
    ```
  }];

  let arguments = (ins I1:$condition, OptionalAttr<StrAttr>:$msg);

  let assemblyFormat = [{ $condition (`,` $msg^)? attr-dict }];
}

// Match format of Index comparisons (for now)
def LLZK_CmpOp : BoolDialectOp<"cmp", [Pure]> {
  let summary = "compare field element values";
  let description = [{
    This operation takes two field element values and compares them according to the
    comparison predicate and returns an `i1`. The following comparisons are supported:

    -   `eq`: equal
    -   `ne`: not equal
    -   `lt`: less than
    -   `le`: less than or equal
    -   `gt`: greater than
    -   `ge`: greater than or equal

    The result is `1` if the comparison is true and `0` otherwise.

    The inequality operators (lt, gt, le, ge) for the finite field elements
    are defined by treating the field elements as integer values:
        `f1 op f2` iff `int(f1) op int(f2)`

    Example:

    ```llzk
    // Less than comparison.
    %0 = bool.cmp lt(%a, %b)

    // Greater than or equal comparison.
    %1 = bool.cmp ge(%a, %b)

    // Not equal comparison.
    %2 = bool.cmp ne(%a, %b)
    ```
  }];

  let arguments = (ins LLZK_CmpPredicateAttr:$predicate, LLZK_FeltType:$lhs,
      LLZK_FeltType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = [{ `` $predicate `(` $lhs `,` $rhs `)` attr-dict }];
}

#endif // LLZK_BOOLEAN_OPS
