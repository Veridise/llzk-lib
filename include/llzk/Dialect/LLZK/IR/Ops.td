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
include "llzk/Dialect/LLZK/IR/Types.td"
include "llzk/Dialect/LLZK/IR/OpsBase.td"
include "llzk/Dialect/Array/IR/Types.td"
include "llzk/Dialect/Function/IR/OpTraits.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

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

def LLZK_ConstReadOp
    : LLZK_Op<
          "read_const", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "read value of a struct parameter";
  let description = [{
    This operation reads the value from the named constant parameter of
    the struct/component in which this op appears. The op itself puts
    some restriction on the type of this value, but leaves it to a later
    type-checking pass to ensure the struct parameters are instantiated
    with types matching the uses of the parameter within the struct.
  }];

  let arguments = (ins FlatSymbolRefAttr:$const_name);
  let results = (outs ConstReadType:$val);

  let assemblyFormat = [{ $const_name `:` type($val) attr-dict }];
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

//===------------------------------------------------------------------===//
// Conversion operators
//===------------------------------------------------------------------===//

def LLZK_UnifiableCastOp : LLZK_Op<"unifiable_cast", [Pure]> {
  let summary = "cast between two unifiable types";
  let description = [{
    This operation reinterprets a value as a different type with the restriction
    that the input and output types of the cast are unifiable.

    Most ops that accept LLZK types accept unifiable types as input and thus there
    is no need for casting between types. This op is meant to be used in situations where
    is not possible to modify the given or the target type and they are different but unifiable.
    For example, inside a conversion pattern the driver may introduce `unrealized_conversion_cast`
    operations if the types are not equal. This will happen regardless of whether the two types unify.
    This cast can be introduced instead of the default cast operation to satisfy MLIR's assumptions
    on type equality.

    Example:
    ```llzk
    %0 = some_other_op : !array.array<@N x !felt.felt>
    %1 = unifiable_cast %0 : (!array.array<@N x @felt.felt>) -> !array.array<affine_map<()[s0, s1] -> (s0 + s1)> x !felt.felt>
    ```
  }];

  let arguments = (ins AnyLLZKType:$input);
  let results = (outs AnyLLZKType:$result);
  let assemblyFormat = [{
    $input `:` functional-type($input, results) attr-dict
  }];

  let hasVerifier = 1;
}

//===------------------------------------------------------------------===//
// Other operators
//===------------------------------------------------------------------===//

def LLZK_ApplyMapOp : LLZK_Op<"applymap", [Pure]> {
  let summary = "apply an AffineMap";
  let description = [{
    This operation applies an AffineMap to a list of SSA values, yielding a single
    SSA value. The number of dimension and symbol arguments must be equal to the
    respective number of dimensional and symbolic inputs to the AffineMap; the
    AffineMap has to be one-dimensional, and so this operation always returns one
    value. The input operands and result all have `index` type.

    Named map example:
    ```llzk
    #map10 = affine_map<(d0, d1) -> (d0 floordiv 8 + d1 floordiv 128)>
    ...
    %1 = llzk.applymap(%s, %t) #map10
    ```

    Inline example:
    ```llzk
    %2 = llzk.applymap(%42)[%n] affine_map<(i)[s0] -> (i+s0)>
    ```
  }];

  let arguments = (ins AffineMapAttr:$map, Variadic<Index>:$mapOperands,
      IndexAttr:$numDims);
  let results = (outs Index);

  // Define builders manually so inference of `numDims` attribute is not
  // circumvented.
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "::mlir::AffineMapAttr":$map,
                                CArg<"::mlir::ValueRange", "{}">:$mapOperands),
                            [{
                  $_state.addOperands(mapOperands);
                  Properties &props = $_state.getOrAddProperties<Properties>();
                  props.setMap(map);
                  props.setNumDims($_builder.getIntegerAttr($_builder.getIndexType(),
                                                            map.getAffineMap().getNumDims()));
                  $_state.addTypes($_builder.getIndexType());
                }]>,
                  OpBuilder<(ins "::mlir::AffineMap":$map,
                                CArg<"::mlir::ValueRange", "{}">:$mapOperands),
                            [{
                  build($_builder, $_state, ::mlir::AffineMapAttr::get(map), mapOperands);
                }]>,
                  OpBuilder<(ins "::mlir::AffineExpr":$expr,
                                CArg<"::mlir::ValueRange", "{}">:$mapOperands),
                            [{
                  auto map = ::mlir::AffineMap::inferFromExprList({expr}).front();
                  build($_builder, $_state, map, mapOperands);
                }]>];

  let assemblyFormat = [{
    custom<DimAndSymbolList>($mapOperands, $numDims) $map attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the affine map to be applied by this operation.
    ::mlir::AffineMap inline getAffineMap() { return getMap(); }

    /// Returns the affine value map computed from this operation.
    ::mlir::affine::AffineValueMap getAffineValueMap() {
      return ::mlir::affine::AffineValueMap(getAffineMap(), getOperands(), getResult());
    }

    /// Returns all dimension operands.
    ::mlir::ValueRange getDimOperands() {
      return ::mlir::OperandRange{
                          getOperands().begin(),
                          getOperands().begin() + getMap().getNumDims()};
    }

    /// Returns all symbol operands.
    ::mlir::ValueRange getSymbolOperands() {
      return ::mlir::OperandRange{
                          getOperands().begin() + getMap().getNumDims(),
                          getOperands().end()};
    }
  }];
}

#endif // LLZK_OPS
