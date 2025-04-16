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

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "llzk/Dialect/LLZK/IR/Attrs.td"
include "llzk/Dialect/LLZK/IR/Types.td"
include "llzk/Dialect/LLZK/IR/OpsBase.td"
include "llzk/Dialect/LLZK/IR/FuncOps.td"

//===------------------------------------------------------------------===//
// Basic constructs
//===------------------------------------------------------------------===//

def LLZK_FeltConstantOp
    : LLZK_Op<"constfelt", [ConstantLike, Pure,
                            DeclareOpInterfaceMethods<
                                OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "field element constant";
  let description = [{
    This operation produces a felt-typed SSA value holding an integer constant.

    Example:

    ```llzk
    %0 = llzk.constfelt 42
    ```
  }];

  let arguments = (ins LLZK_FeltConstAttr:$value);
  let results = (outs LLZK_FeltType:$result);
  let assemblyFormat = [{ $value attr-dict }];
  let hasFolder = 1;
}

def LLZK_FeltNonDetOp
    : LLZK_Op<"nondetfelt", [ConstantLike, Pure,
                             DeclareOpInterfaceMethods<
                                 OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "uninitialized field element";
  let description = [{
    This operation produces a felt-typed SSA value without a specified value.
    This can be used in `constrain()` functions in place of expressions that
    cannot be included in constraints.

    Example:

    ```llzk
    %0 = llzk.nondetfelt
    ```
  }];

  let results = (outs LLZK_FeltType:$result);
  let assemblyFormat = [{ attr-dict }];
}

def LLZK_IncludeOp
    : LLZK_Op<"include", [HasParent<"::mlir::ModuleOp">, Symbol]> {
  let summary = "include operation";
  let description = [{
    This operation imports the contents of another source file in place of itself.

    Example:

    ```llzk
    llzk.include "lib.llzk" as @aliasName
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
      Arg<StrAttr, "source file to include">:$path);

  let assemblyFormat = [{ $path `as` $sym_name attr-dict }];

  let extraClassDeclaration = [{
    static IncludeOp create(::mlir::Location, ::llvm::StringRef name, ::llvm::StringRef path);
    static IncludeOp create(::mlir::Location, ::mlir::StringAttr name, ::mlir::StringAttr path);
    /// Opens the module this include references but doesn't insert it into the parent module.
    ::mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> openModule();
    /// Opens the module this include references and replace this include with that module.
    ::mlir::FailureOr<mlir::ModuleOp> inlineAndErase();
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
    llzk.global const @s : !llzk.string = "Hello World!"

    // Global variable (i.e. no "const" modifier) with initial value.
    llzk.global @b : i1 = false

    // Uninitialized global variable.
    llzk.global @a : !llzk.array<2,2 x i1>
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

def LLZK_GlobalWriteOp
    : LLZK_GlobalRefOp<
          "writeg", [InStructFunctionNamed<"::llzk::FUNC_NAME_COMPUTE">]> {
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
// Struct Operations
//===------------------------------------------------------------------===//

def LLZK_StructDefOp
    : LLZK_Op<"struct", [HasParent<"::mlir::ModuleOp">, Symbol, SymbolTable,
                         IsolatedFromAbove, GraphRegionNoTerminator,
                         OpAsmOpInterface,
                         DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "circuit component definition";
  let description = [{
      This operation describes a component in a circuit. It can contain any number
      of fields that hold inputs, outputs, intermediate values, and subcomponents
      of the defined component. It also contains a `compute()` function that holds
      the witness generation code for the component and a `constrain()` function
      that holds that constraint generation code for the component.

      Example:

      ```llzk
      llzk.struct @ComponentA {
        field @f1 : !llzk.array<5 x index>
        field @f2 : !llzk.felt {llzk.pub}

        func @compute(%p: !llzk.felt) -> !llzk.struct<@ComponentA> {
          %self = new_struct : !llzk.struct<@ComponentA>
          // initialize all fields of `%self` here
          return %self : !llzk.struct<@ComponentA>
        }

        func @constrain(%self: !llzk.struct<@ComponentA>, %p: !llzk.felt) {
          // emit constraints here
          return
        }
      }
      ```
  }];

  // Note: `$const_params` contains symbol definitions that do not use the
  // standard SymbolTable mechanism. Instead hasParamNamed() can be used to
  // check if a certain FlatSymbolRefAttr is a parameter in the function.
  let arguments = (ins SymbolNameAttr:$sym_name,
      OptionalAttr<FlatSymbolRefArrayAttr>:$const_params);

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name (`<` $const_params^ `>`)? $body attr-dict
  }];

  let extraClassDeclaration = [{
    /// Gets the StructType representing this struct. If the `constParams` to use in
    /// the type are not given, the StructType will use `this->getConstParamsAttr()`.
    StructType getType(::std::optional<::mlir::ArrayAttr> constParams = {});

    /// Gets the FieldDefOp that defines the field in this
    /// structure with the given name, if present.
    FieldDefOp getFieldDef(::mlir::StringAttr fieldName);

    /// Get all FieldDefOp in this structure.
    ::std::vector<FieldDefOp> getFieldDefs();

    /// Returns wether the struct defines fields marked as columns.
    mlir::LogicalResult hasColumns() {
      return mlir::success(llvm::any_of(getFieldDefs(), [](FieldDefOp fdOp) {
        return fdOp.getColumn();
      }));
    }

    /// Gets the FuncOp that defines the compute function in this structure, if present.
    FuncOp getComputeFuncOp();

    /// Gets the FuncOp that defines the constrain function in this structure, if present.
    FuncOp getConstrainFuncOp();

    /// Generate header string, in the same format as the assemblyFormat
    ::std::string getHeaderString();

    /// Return `true` iff this StructDefOp has a parameter with the given name
    bool hasParamNamed(::mlir::StringAttr find);
    inline bool hasParamNamed(::mlir::FlatSymbolRefAttr find) {
      return hasParamNamed(find.getRootReference());
    }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static ::llvm::StringRef getDefaultDialect() {
      return "llzk";
    }

    //===------------------------------------------------------------------===//
    // Utility Methods
    //===------------------------------------------------------------------===//

    /// Return the full name for this struct from the root module, including
    /// any surrounding module scopes.
    ::mlir::SymbolRefAttr getFullyQualifiedName();

    /// Return `true` iff this StructDefOp is named "Main".
    bool isMainComponent();
  }];

  let hasRegionVerifier = 1;
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

def LLZK_FieldDefOp
    : LLZK_Op<"field", [HasParent<"::llzk::StructDefOp">,
                        DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                        Symbol]> {
  let summary = "struct field definition";
  let description = [{
    This operation describes a field in a struct/component.

    Example:

    ```llzk
    llzk.field @f1 : !llzk.felt
    llzk.field @f2 : !llzk.felt {llzk.pub}
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<AnyLLZKType>:$type,
      UnitAttr:$column);

  // Define builders manually to avoid the default ones that have extra
  // TypeRange parameters that must always be empty.
  let skipDefaultBuilders = 1;
  let builders =
      [OpBuilder<(ins "::mlir::StringAttr":$sym_name, "::mlir::TypeAttr":$type,
           CArg<"bool", "false">:$isColumn)>,
       OpBuilder<(ins "::llvm::StringRef":$sym_name, "::mlir::Type":$type,
           CArg<"bool", "false">:$isColumn)>,
       OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
           "::mlir::ValueRange":$operands,
           "::llvm::ArrayRef<::mlir::NamedAttribute>":$attributes,
           CArg<"bool", "false">:$isColumn)>,
       // Simpler version since 'resultTypes' and 'operands' must be empty
       OpBuilder<
           (ins "::llvm::ArrayRef<::mlir::NamedAttribute>":$attributes,
               CArg<"bool", "false">:$isColumn),
           [{ build($_builder, $_state, {}, {}, attributes, isColumn); }]>];

  let assemblyFormat = [{ $sym_name `:` $type attr-dict }];

  let extraClassDeclaration = [{
    inline bool hasPublicAttr() { return getOperation()->hasAttr(llzk::PublicAttr::name); }
    void setPublicAttr(bool newValue = true);
  }];
}

def LLZK_FieldReadOp
    : LLZK_FieldRefOp<"readf", [VerifySizesForMultiAffineOps<1>]> {
  let summary = "read value of a struct field";
  let description = [{
    This operation reads the value of a named field in a struct/component.

    The value can be read from the signals table, in which case it can be
    offset by a constant value. A negative value represents reading a value
    backwards and a positive value represents reading a value forward.
    Only fields marked as columns can be read in this manner.
  }];

  let arguments = (ins LLZK_StructType:$component,
      FlatSymbolRefAttr:$field_name,
      OptionalAttr<AnyAttrOf<[SymbolRefAttr, IndexAttr,
                              AffineMapAttr]>>:$tableOffset,
      VariadicOfVariadic<Index, "mapOpGroupSizes">:$mapOperands,
      DefaultValuedAttr<DenseI32ArrayAttr, "{}">:$numDimsPerMap,
      DenseI32ArrayAttr:$mapOpGroupSizes);
  let results = (outs AnyLLZKType:$val);

  // Define builders manually so inference of operand layout attributes is not
  // circumvented.
  let skipDefaultBuilders = 1;
  let builders =
      [OpBuilder<(ins "::mlir::Type":$resultType, "::mlir::Value":$component,
           "::mlir::StringAttr":$field)>,
       OpBuilder<(ins "::mlir::Type":$resultType, "::mlir::Value":$component,
           "::mlir::StringAttr":$field, "::mlir::Attribute":$dist,
           "::mlir::ValueRange":$mapOperands,
           "std::optional<int32_t>":$numDims)>,
       OpBuilder<(ins "::mlir::Type":$resultType, "::mlir::Value":$component,
                     "::mlir::StringAttr":$field,
                     "::mlir::SymbolRefAttr":$dist),
                 [{
              build($_builder, $_state, resultType, component, field, dist, ::mlir::ValueRange(), std::nullopt);
           }]>,
       OpBuilder<(ins "::mlir::Type":$resultType, "::mlir::Value":$component,
                     "::mlir::StringAttr":$field, "::mlir::IntegerAttr":$dist),
                 [{
              build($_builder, $_state, resultType, component, field, dist, ::mlir::ValueRange(), std::nullopt);
           }]>,
       OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
           "::mlir::ValueRange":$operands,
           "::mlir::ArrayRef<::mlir::NamedAttribute>":$attrs)>];

  let assemblyFormat = [{
    $component `[` $field_name `]`
    ( `{` custom<MultiDimAndSymbolList>($mapOperands, $numDimsPerMap)^ `}` )?
    `:` type($component) `,` type($val)
    attr-dict
  }];

  let hasVerifier = 1;
}

def LLZK_FieldWriteOp : LLZK_FieldRefOp<"writef", [ComputeOnly]> {
  let summary = "write value to a struct field";
  let description = [{
    This operation writes a value to a named field in a struct/component.
  }];

  let arguments = (ins LLZK_StructType:$component,
      FlatSymbolRefAttr:$field_name, AnyLLZKType:$val);

  let assemblyFormat = [{
    $component `[` $field_name `]` `=` $val `:` type($component) `,` type($val) attr-dict
  }];
}

def LLZK_CreateStructOp
    : LLZK_Op<"new_struct", [ComputeOnly,
                             DeclareOpInterfaceMethods<
                                 OpAsmOpInterface, ["getAsmResultNames"]>,
                             DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
  let summary = "create a new struct";
  let description = [{
    This operation creates a new, uninitialized instance of a struct.

    Example:

    ```llzk
    %self = llzk.new_struct : !llzk.struct<@Reg>
    ```
  }];

  let results = (outs LLZK_StructType:$result);

  let assemblyFormat = [{ `:` type($result) attr-dict }];
}

//===------------------------------------------------------------------===//
// Array operations
//===------------------------------------------------------------------===//

def LLZK_CreateArrayOp
    : LLZK_Op<
          "new_array",
          [Pure, AttrSizedOperandSegments, VerifySizesForMultiAffineOps<1>,
           DeclareOpInterfaceMethods<SymbolUserOpInterface>,
           DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
           VariadicTypesMatchWith<
               "operand types match result type", "result", "elements",
               "resultTypeToElementsTypes($_self)", "std::equal_to<>()">]> {
  let summary = "create a new array";
  let description = [{
    This operation creates a new array with the given elements.
    The arguments are passed as a flat array but get arranged
    according the shape declared in the type.

    Examples:
    ```llzk
    %0 = llzk.new_array %a, %b, %c : !llzk.array<3 x !llzk.felt>

    // Create an array from the given values using the specified shape
    %1 = llzk.new_array %a, %b, %c, %d : !llzk.array<2,2 x !llzk.felt>

    // Create an uninitialized array
    %2 = llzk.new_array : !llzk.array<3,2 x !llzk.felt>
    ```

    The values used to construct the array must have type that exactly matches
    the element type of the specified array type. This is true even if a `tvar`
    type is used. In other words, cannot mix `tvar<@X>` with `tvar<@Y>` or any
    concrete type. In such a scenario, first create an uninitialized array, as
    shown in the examples above, and then use `writearr` to write each element
    of the array.

    Implementation note: This restriction exists due to a combination of:
    (1) we have chosen to infer the type of `$elements` from the `$result`
    ArrayType, via parseInferredArrayType(), rather than requiring the type of
    every element be listed in the assembly format and,
    (2) within the parser for an Op, there is no way to get the Value instances
    for the operands aside from `OpAsmParser::resolveOperands()` which requires
    the type of every operand to be known and ends up comparing the expected
    to actual type via `operator==`. Thus, there is no way for this to be
    successful apart from all elements having the exact type inferred in (1).

    Also note that `std::equal_to` is used in the `VariadicTypesMatchWith`
    trait on this Op so that the verifier function mirrors the aforementioned
    restriction in the parser.


    In some cases, the length of an uninitialized array depends on the value
    of the loop induction variable (i.e. each iteration creates an array with
    a different size/shape). In that case, an AffineMapAttr can be used to
    specify the dimension size in the ArrayType and the optional instantiation
    parameter list of this operation must be used to instatiate all AffineMap
    used in the array dimensions.

    Examples:
    ```llzk
    // Create an uninitialized array with dimension size defined by AffineMap
    #IdxToLen = affine_map<(i) -> (5*i+1)>
    %3 = llzk.new_array {(%i)} : !llzk.array<#IdxToLen x index>

    // Create an uninitialized array with multiple dimensions defined by
    //  AffineMap. The list of instantiation parameters are assigned to
    //  the AffineMap dimensions left-to-right.
    #M1 = affine_map<(i)[c] -> (c+i)>
    #M3 = affine_map<(m,n) -> (5*m+n)>
    %4 = llzk.new_array{(%i)[%c], (%m,%n)} : !llzk.array<#M1,2,#M3 x i1>
    ```
  }];

  let arguments = (ins Variadic<ArrayElemType>:$elements,
      VariadicOfVariadic<Index, "mapOpGroupSizes">:$mapOperands,
      DefaultValuedAttr<DenseI32ArrayAttr, "{}">:$numDimsPerMap,
      DenseI32ArrayAttr:$mapOpGroupSizes);
  let results = (outs LLZK_ArrayType:$result);

  // Define builders manually so inference of operand layout attributes is not
  // circumvented.
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "::llzk::ArrayType":$result,
                      CArg<"::mlir::ValueRange", "{}">:$elements)>,
                  OpBuilder<(ins "::llzk::ArrayType":$result,
                      "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
                      "::mlir::DenseI32ArrayAttr":$numDimsPerMap)>,
                  OpBuilder<
                      (ins "::llzk::ArrayType":$result,
                          "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
                          "::llvm::ArrayRef<int32_t>":$numDimsPerMap),
                      [{
                        build($_builder, $_state, result, mapOperands, odsBuilder.getDenseI32ArrayAttr(numDimsPerMap));
                      }]>];

  // This uses the custom parseInferredArrayType function to compute the type
  //  of '$elements' to match the type of '$result', except when '$elements'
  //  is empty, then the type of '$elements' must also be empty (size == 0).
  // The if-then-else has '$elements' second so that an empty '$elements' list
  //  can be parsed when neither of these is specified.
  let assemblyFormat = [{
        ( `{` custom<MultiDimAndSymbolList>($mapOperands, $numDimsPerMap)^ `}` ) : ( $elements )?
        `:` type($result)
        `` custom<InferredArrayType>(type($elements), ref($elements), ref(type($result)))
        custom<AttrDictWithWarnings>(attr-dict, prop-dict)
      }];

  let extraClassDeclaration = [{
    static ::mlir::ParseResult parseInferredArrayType(::mlir::OpAsmParser &parser,
        ::llvm::SmallVector<::mlir::Type,1> &elementsTypes,
        ::mlir::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> elements,
        ::mlir::Type resultType
    );
    static void printInferredArrayType(::mlir::OpAsmPrinter &printer, CreateArrayOp,
        ::mlir::TypeRange, ::mlir::OperandRange, ::mlir::Type
    );
  private:
    static ::llvm::SmallVector<::mlir::Type> resultTypeToElementsTypes(::mlir::Type resultType);
  }];

  let hasVerifier = 1;
}

def LLZK_ReadArrayOp
    : LLZK_Op<
          "readarr", [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                      ArrayTypeElemsUnifyWithResultCustomInfer<"arr_ref">]> {
  let summary = "read scalar from an array";
  let description = [{
    This operation reads the value from an array at the specified position.

    Example of 1-dimensional array:
    ```llzk
    %i = arith.constant 0 : index
    %0 = llzk.new_array %a, %b, %c : !llzk.array<3 x !llzk.felt>
    // %1 is now equal to %a
    %1 = llzk.readarr %0[%i] : !llzk.array<3 x !llzk.felt>, !llzk.felt
    ```

    Example of 3-dimensional array:
    ```llzk
    %i = arith.constant 0 : index
    %j = arith.constant 1 : index
    %k = arith.constant 4 : index
    %0 = llzk.new_array ... : !llzk.array<3,10,15 x !llzk.felt>
    // %1 is now equal to %a
    %1 = llzk.readarr %0[%i, %j, %k] : !llzk.array<3,10,15 x !llzk.felt>, !llzk.felt
    ```
  }];

  let arguments = (ins LLZK_ArrayType:$arr_ref, Variadic<Index>:$indices);
  let results = (outs ArrayElemType:$result);

  let assemblyFormat = [{
    $arr_ref `[` $indices `]` `:` type($arr_ref) `,` type($result) attr-dict
  }];
}

def LLZK_WriteArrayOp
    : LLZK_Op<"writearr", [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                           ArrayElemTypeUnifyWith<"arr_ref", "rvalue">]> {
  let summary = "write scalar to an array";
  let description = [{
    This operation writes a value into an array at the specified position.

    Example of 1-dimensional array:
    ```llzk
    %i = arith.constant 0 : index
    %0 = llzk.constfelt 42
    %1 = llzk.new_array %a, %b, %c : !llzk.array<3 x !llzk.felt>
    // The array now is [%0, %b, %c]
    llzk.writearr %1[%i] = %0 : !llzk.array<3 x !llzk.felt>, !llzk.felt
    ```

    Example of 2-dimensional array:
    ```llzk
    %i = arith.constant 0 : index
    %j = arith.constant 0 : index
    %0 = llzk.constfelt 42
    %1 = llzk.new_array %a, %b, %c, %d : !llzk.array<2,2 x !llzk.felt>
    // The array now is [[%0, %b], [%c, %d]]
    llzk.writearr %1[%i, %j] = %0 : !llzk.array<2,2 x !llzk.felt>, !llzk.felt
    ```
  }];

  let arguments = (ins LLZK_ArrayType:$arr_ref, Variadic<Index>:$indices,
      ArrayElemType:$rvalue);

  let assemblyFormat = [{
    $arr_ref `[` $indices `]` `=` $rvalue `:` type($arr_ref) `,` type($rvalue) attr-dict
  }];
}

def LLZK_ExtractArrayOp
    : LLZK_Op<"extractarr", [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                             InferTypeOpAdaptorWithIsCompatible]> {
  let summary = "read subarray from a multi-dimensional array";
  let description = [{
    This operation takes an N-dimensional array and K indices and extracts the
    (N-K)-dimensional array by applying the given indices to the first `K`
    dimensions of the array. Error if `K >= N`. Use `readarr` instead if `K == N`.

    Extracting a 1-D array from 3-D array by selecting the index of 2 dimensions:
    ```llzk
    %i = arith.constant 1 : index
    %0 = llzk.new_array ... : !llzk.array<3,10,15 x !llzk.felt>
    %1 = llzk.extractarr %0[%i,%i] : !llzk.array<3,10,15 x !llzk.felt>, !llzk.array<15 x !llzk.felt>
    ```

    Extracting 1-D arrays for subcomponents:
    ```llzk
    scf.for %iv = %lb to %up step %step {
      %p = llzk.extractarr %in[%iv] : !llzk.array<@N,2 x !llzk.felt>
      %c = llzk.readarr %a[%iv] : !llzk.array<@N x !llzk.struct<@SubC>>, !llzk.struct<@SubC>
      llzk.call @SubC::@constrain(%c, %p) : (!llzk.struct<@SubC>, !llzk.array<2 x !llzk.felt>) -> ()
    }
    ```
  }];

  let arguments = (ins LLZK_ArrayType:$arr_ref, Variadic<Index>:$indices);
  let results = (outs LLZK_ArrayType:$result);

  let assemblyFormat = [{
    $arr_ref `[` $indices `]` `:` type($arr_ref) attr-dict
  }];
}

def LLZK_InsertArrayOp
    : LLZK_Op<"insertarr", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "write subarray into a multi-dimensional array";
  let description = [{
    This operation takes an N-dimensional array, K indices, and an (N+K)-dimensional
    array and inserts the N-dimensional array into the (N+K)-dimensional array at the
    position specified by applying the given indices to the first `K` dimensions of
    the (N+K)-dimensional array. Use `writearr` instead if `N == 0` (LLZK array type
    must have at least 1 dimension so a 0-dimensional array cannot exist anyway).

    Inserting 1-D arrays into a 2-D array:
    ```llzk
    %c = llzk.new_array : !llzk.array<2,3 x index>
    // Array %c is uninitialized [[?, ?, ?], [?, ?, ?]]
    %0 = arith.constant 0 : index
    %a = llzk.new_array %r, %s, %t : !llzk.array<3 x index>
    llzk.insertarr %c[%0] = %a : !llzk.array<2,3 x index>, !llzk.array<3 x index>
    // Array %c is now [[%r, %s, %t], [?, ?, ?]]
    %1 = arith.constant 1 : index
    %b = llzk.new_array %x, %y, %z : !llzk.array<3 x index>
    llzk.insertarr %c[%1] = %b : !llzk.array<2,3 x index>, !llzk.array<3 x index>
    // Array %c is now [[%r, %s, %t], [%x, %y, %z]]
    ```
  }];

  let arguments = (ins LLZK_ArrayType:$arr_ref, Variadic<Index>:$indices,
      LLZK_ArrayType:$rvalue);

  let assemblyFormat = [{
    $arr_ref `[` $indices `]` `=` $rvalue `:` type($arr_ref) `,` type($rvalue) attr-dict
  }];

  let hasVerifier = 1;
}

def LLZK_ArrayLengthOp
    : LLZK_Op<"array_len", [Pure,
                            DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "return the length of an array";
  let description = [{
    This operation returns the size of the specified dimension of an array.

    Example:
    ```llzk
    %a = llzk.new_array : !llzk.array<2,3 x !llzk.felt>
    %0 = arith.constant 0 : index
    %x = llzk.array_len %a, %0 : !llzk.array<2,3 x !llzk.felt> // result is 2
    %1 = arith.constant 1 : index
    %y = llzk.array_len %a, %1 : !llzk.array<2,3 x !llzk.felt> // result is 3
    ```
  }];

  let arguments = (ins LLZK_ArrayType:$arr_ref, Index:$dim);
  let results = (outs Index:$length);

  let assemblyFormat = [{ $arr_ref `,` $dim `:` type($arr_ref) attr-dict }];
}

//===------------------------------------------------------------------===//
// Constraint emit operations
//===------------------------------------------------------------------===//

class LLZK_EmitOp<string mnemonic, list<Trait> traits = []>
    : LLZK_Op<mnemonic,
              traits#[InStructFunctionNamed<"::llzk::FUNC_NAME_CONSTRAIN">,
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

def LLZK_EmitEqualityOp
    : LLZK_EmitOp<"emit_eq", [Commutative, ElementwiseMappable,
                              TypesUnify<"lhs", "rhs">,
]> {
  let summary = "emit an equality constraint";
  let description = [{
    Emits an equality constraint between lhs and rhs. For the built-in Signal
    struct, the constraint is performed on the Signal's inner `@reg` felt field.
  }];

  let arguments = (ins EmitEqType:$lhs, EmitEqType:$rhs);
}

def LLZK_EmitContainmentOp
    : LLZK_EmitOp<"emit_in", [ArrayElemTypeUnifyWith<"lhs", "rhs">]> {
  let summary = "emit a containment constraint";
  let description = [{}];

  let arguments = (ins LLZK_ArrayType:$lhs, ArrayElemType:$rhs);
}

//===------------------------------------------------------------------===//
// Field element operators
//===------------------------------------------------------------------===//

def LLZK_AddFeltOp : LLZK_BinaryOpBase<"add", LLZK_FeltType, [Commutative]> {
  let summary = "addition operator for field elements";
  let description = [{}];
}

def LLZK_SubFeltOp : LLZK_BinaryOpBase<"sub", LLZK_FeltType> {
  let summary = "subtraction operator for field elements";
  let description = [{}];
}

def LLZK_MulFeltOp : LLZK_BinaryOpBase<"mul", LLZK_FeltType, [Commutative]> {
  let summary = "multiplication operator for field elements";
  let description = [{}];
}

def LLZK_DivFeltOp : LLZK_BinaryOpBase<"div", LLZK_FeltType> {
  let summary = "division operator for field elements";
  let description = [{}];
}

def LLZK_ModFeltOp : LLZK_BinaryOpBase<"mod", LLZK_FeltType> {
  let summary = "modulus/remainder operator for field elements";
  let description = [{}];
}

def LLZK_NegFeltOp : LLZK_UnaryOpBase<"neg", LLZK_FeltType> {
  let summary = "negation operator for field elements";
  let description = [{}];
}

def LLZK_InvFeltOp : LLZK_UnaryOpBase<"inv", LLZK_FeltType, [ComputeOnly]> {
  let summary = "inverse operator for field elements";
  let description = [{}];
}

def LLZK_AndFeltOp
    : LLZK_BinaryOpBase<"bit_and", LLZK_FeltType, [ComputeOnly, Commutative]> {
  let summary = "bitwise AND operator for field elements";
  let description = [{}];
}

def LLZK_OrFeltOp
    : LLZK_BinaryOpBase<"bit_or", LLZK_FeltType, [ComputeOnly, Commutative]> {
  let summary = "bitwise OR operator for field elements";
  let description = [{}];
}

def LLZK_XorFeltOp
    : LLZK_BinaryOpBase<"bit_xor", LLZK_FeltType, [ComputeOnly, Commutative]> {
  let summary = "bitwise XOR operator for field elements";
  let description = [{}];
}

def LLZK_NotFeltOp : LLZK_UnaryOpBase<"bit_not", LLZK_FeltType, [ComputeOnly]> {
  let summary = "bit flip operator for field elements";
  let description = [{}];
}

def LLZK_ShlFeltOp : LLZK_BinaryOpBase<"shl", LLZK_FeltType, [ComputeOnly]> {
  let summary = "left shift operator for field elements";
  let description = [{}];
}

def LLZK_ShrFeltOp : LLZK_BinaryOpBase<"shr", LLZK_FeltType, [ComputeOnly]> {
  let summary = "right shift operator for field elements";
  let description = [{}];
}

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

def LLZK_AndBoolOp : LLZK_BinaryOpBase<"and", I1, [ComputeOnly, Commutative]> {
  let summary = "logical AND operator";
  let description = [{
    This operation computes the logical AND (i.e. conjunction) of two `i1` (i.e. boolean)
    values as an `i1` value. The result is `1` if the operation is true and `0` otherwise.
  }];
}

def LLZK_OrBoolOp : LLZK_BinaryOpBase<"or", I1, [ComputeOnly, Commutative]> {
  let summary = "logical OR operator";
  let description = [{
    This operation computes the logical OR (i.e. disjunction) of two `i1` (i.e. boolean)
    values as an `i1` value. The result is `1` if the operation is true and `0` otherwise.
  }];
}

def LLZK_XorBoolOp : LLZK_BinaryOpBase<"xor", I1, [ComputeOnly, Commutative]> {
  let summary = "logical XOR operator";
  let description = [{
    This operation computes the logical XOR (i.e. exclusive disjunction) of two `i1` (i.e. boolean)
    values as an `i1` value. The result is `1` if the operation is true and `0` otherwise.
  }];
}

def LLZK_NotBoolOp : LLZK_UnaryOpBase<"not", I1, [ComputeOnly]> {
  let summary = "logical NOT operator";
  let description = [{
    This operation computes the logical NOT (i.e. negation) of an `i1` (i.e. boolean)
    value as an `i1` value. The result is `1` if the operation is true and `0` otherwise.
  }];
}

//===------------------------------------------------------------------===//
// Conversion operators
//===------------------------------------------------------------------===//

def LLZK_IntToFeltOp : LLZK_Op<"tofelt", [Pure]> {
  let summary = "convert an integer into a field element";
  let description = [{
    This operation converts a supported integer type value into a field element value.

    Example:

    ```llzk
    %0 = llzk.cmp lt(%a, %b)
    %1 = llzk.tofelt %0 : i1
    ```
  }];

  let arguments = (ins AnyLLZKIntType:$value);
  let results = (outs LLZK_FeltType:$result);
  let assemblyFormat = [{ $value `:` type($value) attr-dict }];
}

def LLZK_FeltToIndexOp : LLZK_Op<"toindex", [Pure]> {
  let summary = "convert a field element into an index";
  let description = [{
    This operation converts a field element value into an index value to allow
    use as an array index or loop bound. In struct @constrain functions, the
    argument to this op is not allowed to be derived from a Signal struct.

    Example:
    ```llzk
    %0 = llzk.toindex %a
    %1 = llzk.readarr %b[%0]
    ```
  }];

  let arguments = (ins LLZK_FeltType:$value);
  let results = (outs Index:$result);
  let assemblyFormat = [{ $value attr-dict }];

  let hasVerifier = 1;
}

def LLZK_UnifiableCastOp : LLZK_Op<"unifiable_cast", [Pure]> {
  let summary = "cast between two unifiable types";
  let description = [{
    This operation reinterprets a value as a different type with the restriction
    that the input and output types of the cast are unifiable.

    Most ops that accept LLZK types accept unifiable types as input and thus there
    is no need for casting between types. This op is meant to be used in situations where
    is not possible to modify the given or the target type and they are different but unifiable.
    For example, inside a conversion pattern the driver may introduce `unrealized_conversion_cast`
    operations if the types are not equal. This will happen regardless of wether the two types unify.
    This cast can be introduced instead of the default cast operation to satisfy MLIR's assumtions
    on type equality.

    Example:
    ```llzk
    %0 = some_other_op : !llzk.array<@N x !llzk.felt>
    %1 = unifiable_cast %0 : (!llzk.array<@N x @llzk.felt>) -> !llzk.array<affine_map<()[s0, s1] -> (s0 + s1)> x !llzk.felt>
    ```
  }];

  let arguments = (ins AnyLLZKType:$input);
  let results = (outs AnyLLZKType:$result);
  let assemblyFormat = "$input `:` functional-type($input, results) attr-dict";

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

def LLZK_LitStringOp : LLZK_Op<"new_string", [Pure, ConstantLike]> {
  let summary = "literal string";
  let arguments = (ins StrAttr:$value);
  let results = (outs LLZK_String:$result);

  let assemblyFormat = [{
    $value attr-dict
  }];
  let hasFolder = 1;
}

#endif // LLZK_OPS
