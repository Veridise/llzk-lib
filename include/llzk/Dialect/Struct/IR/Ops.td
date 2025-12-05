//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
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

#ifndef LLZK_STRUCT_OPS
#define LLZK_STRUCT_OPS

include "llzk/Dialect/Function/IR/OpTraits.td"
include "llzk/Dialect/Struct/IR/Dialect.td"
include "llzk/Dialect/Struct/IR/OpInterfaces.td"
include "llzk/Dialect/Struct/IR/Types.td"
include "llzk/Dialect/Shared/OpTraits.td"

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class StructDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<StructDialect, mnemonic, traits>;

/// Only valid/implemented for StructDefOp. Sets the proper `AllowWitnessAttr`
/// and `AllowConstraintAttr` on the functions defined within the StructDefOp.
def SetFuncAllowAttrs : NativeOpTrait<"SetFuncAllowAttrs">, StructuralOpTrait {
  string cppNamespace = "::llzk::component";
}

//===------------------------------------------------------------------===//
// Struct Operations
//===------------------------------------------------------------------===//

def LLZK_StructDefOp
    : StructDialectOp<
          "def", [HasParent<"::mlir::ModuleOp">, Symbol, SymbolTable,
                  IsolatedFromAbove, SetFuncAllowAttrs,
                  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                  NoRegionArguments]#GraphRegionNoTerminator.traits> {
  let summary = "circuit component definition";
  let description = [{
      This operation describes a component in a circuit. It can contain any number
      of fields that hold inputs, outputs, intermediate values, and subcomponents
      of the defined component. It also contains a `compute()` function that holds
      the witness generation code for the component and a `constrain()` function
      that holds that constraint generation code for the component.

      Example:

      ```llzk
      struct.def @ComponentA {
        field @f1 : !array.type<5 x index>
        field @f2 : !felt.type {llzk.pub}

        function.def @compute(%p: !felt.type) -> !struct.type<@ComponentA> {
          %self = struct.new : !struct.type<@ComponentA>
          // initialize all fields of `%self` here
          return %self : !struct.type<@ComponentA>
        }

        function.def @constrain(%self: !struct.type<@ComponentA>, %p: !felt.type) {
          // emit constraints here
          return
        }
      }
      ```

      A special component named `Main` can be defined to represent the main entry point
      of the circuit. This component has additional restrictions:
      1. It cannot have struct parameters.
      2. The parameter types of its functions (besides the required "self" parameter) can
         only be `!felt.type` or `!array.type<.. x !felt.type>`.
      3. The fields marked as public (i.e., with the `{llzk.pub}` attribute) represent
         the public outputs of the overall circuit.

      Example of a `Main` component:

      ```llzk
      struct.def @Main {
        field @out : !felt.type {llzk.pub} // public output
        field @intermediate : !felt.type
        function.def @compute(%p: !felt.type) -> !struct.type<@Main> {
          // ...
        }
        function.def @constrain(%self: !struct.type<@Main>, %p: !felt.type) {
          // ...
        }
      }
      ```
  }];

  // Note: `$const_params` contains symbol definitions that do not use the
  // standard SymbolTable mechanism. Instead hasParamNamed() can be used to
  // check if a certain FlatSymbolRefAttr is a parameter in the function.
  let arguments = (ins SymbolNameAttr:$sym_name,
      OptionalAttr<FlatSymbolRefArrayAttr>:$const_params);

  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    $sym_name (`<` $const_params^ `>`)? $bodyRegion attr-dict
  }];

  let extraClassDeclaration = [{
    /// Gets the StructType representing this struct. If the `constParams` to use in
    /// the type are not given, the StructType will use `this->getConstParamsAttr()`.
    StructType getType(::std::optional<::mlir::ArrayAttr> constParams = {});
    inline StructType getType(::std::optional<::mlir::ArrayAttr> constParams = {}) const {
      return const_cast<StructDefOp*>(this)->getType(constParams);
    }

    /// Gets the FieldDefOp that defines the field in this
    /// structure with the given name, if present.
    FieldDefOp getFieldDef(::mlir::StringAttr fieldName);

    /// Get all FieldDefOp in this structure.
    ::std::vector<FieldDefOp> getFieldDefs();

    /// Returns whether the struct defines fields marked as columns.
    ::mlir::LogicalResult hasColumns() {
      return ::mlir::success(::llvm::any_of(getFieldDefs(), [](FieldDefOp fdOp) {
        return fdOp.getColumn();
      }));
    }

    /// Returns whether the struct defines fields marked as signals.
    ::mlir::LogicalResult hasSignals() {
      return ::mlir::success(::llvm::any_of(getFieldDefs(), [](FieldDefOp fdOp) {
        return fdOp.getSignal();
      }));
    }


    /// Gets the FuncDefOp that defines the compute function in this structure, if present, or `nullptr` otherwise.
    ::llzk::function::FuncDefOp getComputeFuncOp();

    /// Gets the FuncDefOp that defines the constrain function in this structure, if present, or `nullptr` otherwise.
    ::llzk::function::FuncDefOp getConstrainFuncOp();

    /// Gets the FuncDefOp that defines the compute function in this structure, if present, or the product function otherwise.
    ::llzk::function::FuncDefOp getComputeOrProductFuncOp();

    /// Gets the FuncDefOp that defines the constrain function in this structure, if present, or the product function otherwise.
    ::llzk::function::FuncDefOp getConstrainOrProductFuncOp();

    /// Returns `true` iff this structure defines compute and constrain functions.
    bool hasComputeConstrain() { return lookupSymbol(FUNC_NAME_COMPUTE) != nullptr && lookupSymbol(FUNC_NAME_CONSTRAIN) != nullptr; }

    /// Generate header string, in the same format as the assemblyFormat
    ::std::string getHeaderString();

    /// Return `false` iff `getConstParamsAttr()` returns `nullptr`
    bool hasConstParamsAttr() { return getProperties().const_params != nullptr; };

    /// Return `true` iff this StructDefOp has a parameter with the given name
    bool hasParamNamed(::mlir::StringAttr find);
    inline bool hasParamNamed(::mlir::FlatSymbolRefAttr find) {
      return hasParamNamed(find.getRootReference());
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

def LLZK_FieldDefOp
    : StructDialectOp<
          "field", [HasParent<"::llzk::component::StructDefOp">,
                    DeclareOpInterfaceMethods<SymbolUserOpInterface>, Symbol]> {
  let summary = "struct field definition";
  let description = [{
    This operation describes a field in a struct/component.

    Example:

    ```llzk
    struct.field @f1 : !felt.type
    struct.field @f2 : !felt.type {llzk.pub}
    struct.field @col1 : !felt.type {column}
    struct.field @sig1 : !felt.type {signal}
    struct.field @colsig1 : !felt.type {column, signal}
    struct.field @pubcol : !felt.type {llzk.pub, column}
    ```

    - Fields marked with the `{llzk.pub}` attribute are considered public
      and represent outputs of their defining struct/component.
    - Fields marked with the `{column}` attribute can be read/written with
      table offsets using the `readf` and `writef` operations.
    - Fields marked with the `{signal}` attribute are constraint variables that
      are stored in the witness; non-signal values are intermediate expressions.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<AnyLLZKType>:$type,
      UnitAttr:$column, UnitAttr:$signal);

  // Define builders manually to avoid the default ones that have extra
  // TypeRange parameters that must always be empty.
  let skipDefaultBuilders = 1;
  let builders =
      [OpBuilder<(ins "::mlir::StringAttr":$sym_name, "::mlir::TypeAttr":$type,
           CArg<"bool", "false">:$isColumn, CArg<"bool", "false">:$isSignal)>,
       OpBuilder<(ins "::llvm::StringRef":$sym_name, "::mlir::Type":$type,
           CArg<"bool", "false">:$isColumn, CArg<"bool", "false">:$isSignal)>,
       OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
           "::mlir::ValueRange":$operands,
           "::llvm::ArrayRef<::mlir::NamedAttribute>":$attributes,
           CArg<"bool", "false">:$isColumn, CArg<"bool", "false">:$isSignal)>,
       // Simpler version since 'resultTypes' and 'operands' must be empty
       OpBuilder<
           (ins "::llvm::ArrayRef<::mlir::NamedAttribute>":$attributes,
               CArg<"bool", "false">:$isColumn,
               CArg<"bool", "false">:$isSignal),
           [{ build($_builder, $_state, {}, {}, attributes, isColumn, isSignal); }]>];

  let assemblyFormat = [{ $sym_name `:` $type attr-dict }];

  let extraClassDeclaration = [{
    inline bool hasPublicAttr() { return getOperation()->hasAttr(llzk::PublicAttr::name); }
    void setPublicAttr(bool newValue = true);
  }];
}

class FieldRefOpBase<string mnemonic, list<Trait> traits = []>
    : StructDialectOp<
          mnemonic, traits#[DeclareOpInterfaceMethods<FieldRefOpInterface>,
                            DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  bit isRead = ?; // read(1) vs write(0) ops
  let extraClassDeclaration = [{
    /// Gets the definition for the `field` referenced in this op.
    inline ::mlir::FailureOr<SymbolLookupResult<FieldDefOp>> getFieldDefOp(::mlir::SymbolTableCollection &tables) {
      return ::llvm::cast<FieldRefOpInterface>(getOperation()).getFieldDefOp(tables);
    }
  }];
  let extraClassDefinition = [{
    /// Return `true` if the op is a read, `false` if it's a write.
    bool $cppClass::isRead() {
      return }]#!if(isRead, "true", "false")#[{;
    }
  }];
}

def LLZK_FieldReadOp
    : FieldRefOpBase<"readf", [VerifySizesForMultiAffineOps<1>]> {
  let summary = "read value of a struct field";
  let description = [{
    This operation reads the value of a named field in a struct/component.

    The value can be read from the signals table, in which case it can be
    offset by a constant value. A negative value represents reading a value
    backwards and a positive value represents reading a value forward.
    Only fields marked as columns can be read in this manner.
  }];
  let isRead = 1;

  // See `VerifySizesForMultiAffineOps` for more explanation of these arguments.
  let arguments = (ins LLZK_StructType:$component,
      FlatSymbolRefAttr:$field_name,
      OptionalAttr<AnyAttrOf<[SymbolRefAttr, IndexAttr,
                              AffineMapAttr]>>:$tableOffset,
      // List of AffineMap operand groups where each group provides the
      // arguments to instantiate the next (left-to-right) AffineMap used in
      // `tableOffset`.
      VariadicOfVariadic<Index, "mapOpGroupSizes">:$mapOperands,
      // Within each group in '$mapOperands', denotes the number of values that
      // are AffineMap "dimensional" arguments with the remaining values being
      // AffineMap "symbolic" arguments.
      DefaultValuedAttr<DenseI32ArrayAttr, "{}">:$numDimsPerMap,
      // Denotes the size of each variadic group in '$mapOperands'.
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

def LLZK_FieldWriteOp : FieldRefOpBase<"writef", [WitnessGen]> {
  let summary = "write value to a struct field";
  let description = [{
    This operation writes a value to a named field in a struct/component.
  }];
  let isRead = 0;

  let arguments = (ins LLZK_StructType:$component,
      FlatSymbolRefAttr:$field_name, AnyLLZKType:$val);

  let assemblyFormat = [{
    $component `[` $field_name `]` `=` $val `:` type($component) `,` type($val) attr-dict
  }];
}

def LLZK_CreateStructOp
    : StructDialectOp<"new", [DeclareOpInterfaceMethods<
                                  OpAsmOpInterface, ["getAsmResultNames"]>,
                              DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                              WitnessGen,
]> {
  let summary = "create a new struct";
  let description = [{
    This operation creates a new, uninitialized instance of a struct.

    Example:

    ```llzk
    %self = struct.new : !struct.type<@Reg>
    ```
  }];

  let results = (outs LLZK_StructType:$result);

  let assemblyFormat = [{ `:` type($result) attr-dict }];
}

#endif // LLZK_STRUCT_OPS
