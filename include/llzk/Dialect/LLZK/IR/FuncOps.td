//===-- FuncOps.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from mlir/include/mlir/Dialect/Func/IR/FuncOps.td
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_FUNC_OPS
#define LLZK_FUNC_OPS

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "llzk/Dialect/LLZK/IR/OpsBase.td"

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp
    : LLZK_Op<"func", [ParentOneOf<["::mlir::ModuleOp", "::llzk::StructDefOp"]>,
                       DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                       AffineScope, AutomaticAllocationScope,
                       FunctionOpInterface, IsolatedFromAbove,
                       OpAsmOpInterface]> {
  // NOTE: Cannot have SymbolTable trait because that would cause global
  // functions without a body to produce "Operations with a 'SymbolTable' must
  // have exactly one block"
  let summary = "An operation with a name containing a single `SSACFG` region";
  let description = [{
    Operations within the function cannot implicitly capture values defined
    outside of the function, i.e. Functions are `IsolatedFromAbove`. All
    external references must use function arguments or attributes that establish
    a symbolic connection (e.g. symbols referenced by name via a string
    attribute like SymbolRefAttr). An external function declaration (used when
    referring to a function declared in some other module) has no body. While
    the MLIR textual form provides a nice inline syntax for function arguments,
    they are internally represented as “block arguments” to the first block in
    the region.

    Only dialect attribute names may be specified in the attribute dictionaries
    for function arguments, results, or the function itself.

    Example:

    ```llzk
    // External function definitions.
    llzk.func private @abort()
    llzk.func private @scribble(!llzk.array<5 x !llzk.felt>, !llzk.struct<@Hello>) -> i1

    // A function that returns its argument twice:
    llzk.func @count(%x: !llzk.felt) -> (!llzk.felt, !llzk.felt) {
      return %x, %x: !llzk.felt, !llzk.felt
    }

    // Function definition within a component
    llzk.struct @NonZero {
      func @compute(%a: !llzk.felt) { return }
      func @constrain(%a: !llzk.felt) { return }
    }
    ```
  }];

  // Duplicated from the pre-defined `func` dialect. We don't store the
  // visibility attribute but, since we use `function_interface_impl` for
  // parsing/printing, there is still the requirement that global functions
  // declared without a body must specify the `private` visibility.
  // Additionally, the default parsing/printing functions allow attributes on
  // the arguments, results, and function itself.
  //    ```llzk
  //    // Argument attribute
  //    llzk.func private @example_fn_arg(%x: i1 {llzk.pub})
  //
  //    // Result attribute
  //    llzk.func @example_fn_result() -> (i1 {dialectName.attrName = 0 : i1})
  //
  //    // Function attribute
  //    llzk.func @example_fn_attr() attributes {dialectName.attrName = false}
  //    ```
  let arguments = (ins SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins "::llvm::StringRef":$name,
      "::mlir::FunctionType":$type,
      CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs,
      CArg<"::llvm::ArrayRef<::mlir::DictionaryAttr>", "{}">:$argAttrs)>];
  let extraClassDeclaration = [{
    static FuncOp create(::mlir::Location location, ::llvm::StringRef name, ::mlir::FunctionType type,
                         ::llvm::ArrayRef<::mlir::NamedAttribute> attrs = {});
    static FuncOp create(::mlir::Location location, ::llvm::StringRef name, ::mlir::FunctionType type,
                         ::mlir::Operation::dialect_attr_range attrs);
    static FuncOp create(::mlir::Location location, ::llvm::StringRef name, ::mlir::FunctionType type,
                         ::llvm::ArrayRef<::mlir::NamedAttribute> attrs,
                         ::llvm::ArrayRef<::mlir::DictionaryAttr> argAttrs);

    /// Create a deep copy of this function and all of its blocks, remapping any
    /// operands that use values outside of the function using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for function arguments, these arguments are not
    /// included in the new function. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    FuncOp clone(::mlir::IRMapping &mapper);
    FuncOp clone();

    /// Clone the internal blocks and attributes from this function into dest.
    /// Any cloned blocks are appended to the back of dest. This function
    /// asserts that the attributes of the current function and dest are
    /// compatible.
    void cloneInto(FuncOp dest, ::mlir::IRMapping &mapper);

    /// Return `true` iff the argument at the given index has `pub` attribute.
    bool hasArgPublicAttr(unsigned index);

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the argument types of this function.
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static ::llvm::StringRef getDefaultDialect() { return "llzk"; }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }

    //===------------------------------------------------------------------===//
    // Utility Methods
    //===------------------------------------------------------------------===//

    /// Return the full name for this function from the root module, including
    /// all module and struct scopes.
    ::mlir::SymbolRefAttr getFullyQualifiedName();

    /// Return `true` iff the function name is `FUNC_NAME_COMPUTE` (if needed, a check
    /// that this FuncOp is located within a StructDefOp must be done separately).
    inline bool nameIsCompute() { return FUNC_NAME_COMPUTE == getSymName(); }

    /// Return `true` iff the function name is `FUNC_NAME_CONSTRAIN` (if needed, a
    /// check that this FuncOp is located within a StructDefOp must be done separately).
    inline bool nameIsConstrain() { return FUNC_NAME_CONSTRAIN == getSymName(); }

    /// Return `true` iff the function is with a StructDefOp
    bool isInStruct() { return ::llzk::isInStruct(*this); }

    /// Return `true` iff the function is with a StructDefOp and named `FUNC_NAME_COMPUTE`.
    inline bool isStructCompute() { return isInStruct() && nameIsCompute(); }

    /// Return `true` iff the function is with a StructDefOp and named `FUNC_NAME_CONSTRAIN`.
    inline bool isStructConstrain() { return isInStruct() && nameIsConstrain(); }

    /// Assuming the name is `FUNC_NAME_COMPUTE`, return the single StructType result.
    ::llzk::StructType getComputeSingleResultType();
  }];
  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp
    : LLZK_Op<"return", [HasParent<"::llzk::FuncOp">, Pure, MemRefsNormalizable,
                         ReturnLike, Terminator]> {
  let summary = "Function return operation";
  let description = [{
    The `llzk.return` operation represents a return operation within a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation.

    Example:

    ```llzk
    llzk.func @foo() : (!llzk.felt, index) {
      ...
      return %0, %1 : !llzk.felt, index
    }
    ```
  }];

  let arguments = (ins Variadic<AnyLLZKType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def CallOp
    : LLZK_Op<"call", [MemRefsNormalizable, AttrSizedOperandSegments,
                       VerifySizesForMultiAffineOps<1>,
                       DeclareOpInterfaceMethods<CallOpInterface>,
                       DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call operation";
  let description = [{
    The `llzk.call` operation represents a call to another function. The operands
    and result types of the call must match the specified function type. The
    callee is encoded as a symbol reference attribute named "callee" which must
    be the full path to the target function from the root module (i.e. the module
    containing the [llzk::LANG_ATTR_NAME] attribute).

    Example:
    ```llzk
    // Call a global function defined in the root module.
    llzk.call @do_stuff(%0) : (!llzk.struct<@Bob>) -> ()
    %1, %2 = llzk.call @split(%x) : (index) -> (index, index)

    // Call a function within a component
    %2 = llzk.call @OtherStruct::@compute(%3, %4) : (index, index) -> !llzk.struct<@OtherStruct>
    llzk.call @OtherStruct::@constrain(%5, %6) : (!llzk.struct<@OtherStruct>, !llzk.felt) -> ()
    ```

    When the return StructType of a `compute()` function uses AffineMapAttr to
    express struct parameter(s) that depend on a loop variable, the optional
    instantiation parameter list of this operation must be used to instatiate
    all AffineMap used as parameters to the StructType.

    Examples:
    ```llzk
    #M = affine_map<(i)[] -> (5*i+1)>
    %r = llzk.call @A::@compute(%x){(%i)} : (!llzk.felt) -> !llzk.struct<@A<[#M]>>
    ```
  }];

  let arguments = (ins SymbolRefAttr:$callee,
      Variadic<AnyLLZKType>:$argOperands,
      VariadicOfVariadic<Index, "mapOpGroupSizes">:$mapOperands,
      DefaultValuedAttr<DenseI32ArrayAttr, "{}">:$numDimsPerMap,
      DenseI32ArrayAttr:$mapOpGroupSizes);
  let results = (outs Variadic<AnyLLZKType>);

  // Define builders manually so inference of operand layout attributes is not
  // circumvented.
  let skipDefaultBuilders = 1;
  let builders =
      [OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
           "::mlir::SymbolRefAttr":$callee,
           CArg<"::mlir::ValueRange", "{}">:$argOperands)>,
       OpBuilder<(ins "::llzk::FuncOp":$callee,
                     CArg<"::mlir::ValueRange", "{}">:$argOperands),
                 [{
                    build($_builder, $_state, callee.getFunctionType().getResults(),
                          ::mlir::SymbolRefAttr::get(callee), argOperands);
                  }]>,
       OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
           "::mlir::SymbolRefAttr":$callee,
           "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
           "::mlir::DenseI32ArrayAttr":$numDimsPerMap,
           CArg<"::mlir::ValueRange", "{}">:$argOperands)>,
       OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
                     "::mlir::SymbolRefAttr":$callee,
                     "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
                     "::llvm::ArrayRef<int32_t>":$numDimsPerMap,
                     CArg<"::mlir::ValueRange", "{}">:$argOperands),
                 [{
                    build($_builder, $_state, resultTypes, callee, mapOperands,
                          $_builder.getDenseI32ArrayAttr(numDimsPerMap), argOperands);
                  }]>,
       OpBuilder<(ins "::llzk::FuncOp":$callee,
                     "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
                     "::mlir::DenseI32ArrayAttr":$numDimsPerMap,
                     CArg<"::mlir::ValueRange", "{}">:$argOperands),
                 [{
                    build($_builder, $_state, callee.getFunctionType().getResults(),
                          ::mlir::SymbolRefAttr::get(callee), mapOperands, numDimsPerMap, argOperands);
                  }]>,
       OpBuilder<(ins "::llzk::FuncOp":$callee,
                     "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
                     "::llvm::ArrayRef<int32_t>":$numDimsPerMap,
                     CArg<"::mlir::ValueRange", "{}">:$argOperands),
                 [{
                    build($_builder, $_state, callee, mapOperands,
                          $_builder.getDenseI32ArrayAttr(numDimsPerMap), argOperands);
                  }]>];

  let extraClassDeclaration = [{
    ::mlir::FunctionType getCalleeType();

    /// Return `true` iff the callee function name is `FUNC_NAME_COMPUTE` (this
    /// does not check if the callee function is located within a StructDefOp).
    inline bool calleeIsCompute() { return FUNC_NAME_COMPUTE == getCallee().getLeafReference(); }

    /// Return `true` iff the callee function name is `FUNC_NAME_CONSTRAIN` (this
    /// does not check if the callee function is located within a StructDefOp).
    inline bool calleeIsConstrain() { return FUNC_NAME_CONSTRAIN == getCallee().getLeafReference(); }

    /// Return `true` iff the callee function name is `FUNC_NAME_COMPUTE` within a StructDefOp.
    bool calleeIsStructCompute();

    /// Return `true` iff the callee function name is `FUNC_NAME_CONSTRAIN` within a StructDefOp.
    bool calleeIsStructConstrain();

    /// Assuming the callee is `FUNC_NAME_COMPUTE`, return the single StructType result.
    ::llzk::StructType getComputeSingleResultType();
  }];

  let assemblyFormat = [{
    $callee `(` $argOperands `)`
    ( `{` custom<MultiDimAndSymbolList>($mapOperands, $numDimsPerMap)^ `}` )?
    `:` functional-type($argOperands, results)
    custom<AttrDictWithWarnings>(attr-dict, prop-dict)
  }];

  // NOTE: In CreateArrayOp, the `verify()` function is declared in order to
  // call `verifyAffineMapInstantiations()`. However, in this op that check must
  // happen within `verifySymbolUses()` instead because the target FuncOp must
  // be resolved to determine if a target function named "compute"/"constrain"
  // is defined within a StructDefOp or within a ModuleOp because the
  // verification differs for those cases.
}

#endif // LLZK_FUNC_OPS
