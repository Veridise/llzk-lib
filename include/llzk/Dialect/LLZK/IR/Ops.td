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
include "llzk/Dialect/LLZK/IR/OpInterfaces.td"
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

//===------------------------------------------------------------------===//
// Basic constructs
//===------------------------------------------------------------------===//

def LLZK_UndefOp : LLZK_Op<"undef", [Pure]> {
  let summary = "Creates an undefined value of the specified type.";
  let description = [{
    This operation has no operands or attributes. It represents an undefined
    value of the specified LLZK IR dialect type. This op can be introduced by
    the `llzk-array-to-scalar` pass if there is a read from an array index
    that was not dominated by an earlier write to that same index.

    Example:

    ```llzk
    %0 = llzk.undef : !felt.felt
    ```
  }];
  let results = (outs AnyLLZKType:$res);
  let assemblyFormat = [{
    `:` type($res) attr-dict
  }];
}

def LLZK_AssertOp
    : LLZK_Op<"assert", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "assertion operation";
  let description = [{
    This operation asserts that a given boolean value is true. Assertions are checked
    statically when possible. If the condition evaluates to `true`, the assertion is
    removed. If `false`, an error is reported. Otherwise, the assertion is preserved.
    All assertions that appear in `constrain()` functions must evaluate statically
    (i.e. they cannot depend on inputs to the circuit) else an error is reported.

    Assertion without message:
    ```llzk
    %1 = llzk.cmp lt(%a, %b)
    llzk.assert %1
    ```

    Assertion with a message:
    ```llzk
    %1 = llzk.cmp eq(%a, %b)
    llzk.assert %1, "expected equal values"
    ```
  }];

  let arguments = (ins I1:$condition, OptionalAttr<StrAttr>:$msg);

  let assemblyFormat = [{ $condition (`,` $msg^)? attr-dict }];
}

//===------------------------------------------------------------------===//
// Global storage
//===------------------------------------------------------------------===//

def LLZK_GlobalDefOp
    : LLZK_Op<"global", [HasParent<"mlir::ModuleOp">,
                         DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                         Symbol]> {
  let summary = "global value";
  let description = [{
    Examples:

    ```llzk
    // Global constant (denoted by "const" modifier) string.
    llzk.global const @s : !string.string = "Hello World!"

    // Global variable (i.e. no "const" modifier) with initial value.
    llzk.global @b : i1 = false

    // Uninitialized global variable.
    llzk.global @a : !array.array<2,2 x i1>
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, UnitAttr:$constant,
      TypeAttrOf<GlobalDefType>:$type,
      DefaultValuedAttr<AnyAttr, "nullptr">:$initial_value);

  let assemblyFormat = [{
    (`const` $constant^)?
    $sym_name `:` $type
    `` custom<GlobalInitialValue>($initial_value, ref($type))
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static ::mlir::ParseResult parseGlobalInitialValue(::mlir::OpAsmParser &parser,
        ::mlir::Attribute &initialValue, ::mlir::TypeAttr typeAttr
    );
    static void printGlobalInitialValue(::mlir::OpAsmPrinter &printer, GlobalDefOp op,
        ::mlir::Attribute initialValue, ::mlir::TypeAttr typeAttr
    );

    inline bool isConstant() { return getConstant(); }
  }];
}

def LLZK_GlobalReadOp : LLZK_GlobalRefOp<"readg"> {
  let summary = "read value of a global";
  let description = [{
    This operation reads the value of a named global.
  }];

  let arguments = (ins SymbolRefAttr:$name_ref);
  let results = (outs GlobalDefType:$val);

  let assemblyFormat = [{
    $name_ref `:` type($val) attr-dict
  }];
}

def LLZK_GlobalWriteOp : LLZK_GlobalRefOp<"writeg", [WitnessGen]> {
  let summary = "write value to a global";
  let description = [{
    This operation writes a value to a named global.
    Not allowed for globals declared with the "const" modifier.
  }];

  let arguments = (ins SymbolRefAttr:$name_ref, GlobalDefType:$val);

  let assemblyFormat = [{
    $name_ref `=` $val `:` type($val) attr-dict
  }];
}

//===------------------------------------------------------------------===//
// Field element operators
//===------------------------------------------------------------------===//

// Match format of Index comparisons (for now)
def LLZK_CmpOp : LLZK_Op<"cmp", [Pure]> {
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
    %0 = llzk.cmp lt(%a, %b)

    // Greater than or equal comparison.
    %1 = llzk.cmp ge(%a, %b)

    // Not equal comparison.
    %2 = llzk.cmp ne(%a, %b)
    ```
  }];

  let arguments = (ins LLZK_CmpPredicateAttr:$predicate, LLZK_FeltType:$lhs,
      LLZK_FeltType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = [{ `` $predicate `(` $lhs `,` $rhs `)` attr-dict }];
}

//===------------------------------------------------------------------===//
// Boolean operators
//===------------------------------------------------------------------===//

def LLZK_AndBoolOp : LLZK_BinaryOpBase<"and", I1, [WitnessGen, Commutative]> {
  let summary = "logical AND operator";
  let description = [{
    This operation computes the logical AND (i.e. conjunction) of two `i1` (i.e. boolean)
    values as an `i1` value. The result is `1` if the operation is true and `0` otherwise.
  }];
}

def LLZK_OrBoolOp : LLZK_BinaryOpBase<"or", I1, [WitnessGen, Commutative]> {
  let summary = "logical OR operator";
  let description = [{
    This operation computes the logical OR (i.e. disjunction) of two `i1` (i.e. boolean)
    values as an `i1` value. The result is `1` if the operation is true and `0` otherwise.
  }];
}

def LLZK_XorBoolOp : LLZK_BinaryOpBase<"xor", I1, [WitnessGen, Commutative]> {
  let summary = "logical XOR operator";
  let description = [{
    This operation computes the logical XOR (i.e. exclusive disjunction) of two `i1` (i.e. boolean)
    values as an `i1` value. The result is `1` if the operation is true and `0` otherwise.
  }];
}

def LLZK_NotBoolOp : LLZK_UnaryOpBase<"not", I1, [WitnessGen]> {
  let summary = "logical NOT operator";
  let description = [{
    This operation computes the logical NOT (i.e. negation) of an `i1` (i.e. boolean)
    value as an `i1` value. The result is `1` if the operation is true and `0` otherwise.
  }];
}

#endif // LLZK_OPS
