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
    The `llzk.constfelt` operation produces a felt-typed SSA value equal to
    some felt-typed integer constant.

    Example:

    ```llzk
    %0 = llzk.constfelt 42
    ```
  }];

  let arguments = (ins LLZK_FeltConstAttr:$value);
  let results = (outs LLZK_FeltType:$result);
  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
}

def LLZK_FeltNonDetOp
    : LLZK_Op<"nondetfelt", [ConstantLike, Pure,
                             DeclareOpInterfaceMethods<
                                 OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "uninitialized field element";
  let description = [{
    The `llzk.nondetfelt` operation produces a felt-typed SSA value. This
    can be used in `constrain()` functions in place of expressions that
    cannot be included in constraints.

    Example:

    ```llzk
    %0 = llzk.nondetfelt
    ```
  }];

  let results = (outs LLZK_FeltType:$result);
  let assemblyFormat = "attr-dict";
}

def LLZK_IncludeOp
    : LLZK_Op<"include", [HasParent<"::mlir::ModuleOp">, Symbol]> {
  let summary = "Include operation";
  let description = [{
    The `include` operation allows to import the contents of another source file.

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
  let summary = "Assertion operation";
  let description = [{
    Assertions are checked statically when possible. If the condition evaluates to
    `true`, the assertion is removed. If `false`, an error is reported. Otherwise,
    the assertion is preserved. All assertions that appear in `constrain()` functions
    must evaluate statically, i.e. they cannot depend on inputs to the circuit.

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
// Struct Operations
//===------------------------------------------------------------------===//

def LLZK_StructDefOp
    : LLZK_Op<"struct", [HasParent<"::mlir::ModuleOp">, Symbol, SymbolTable,
                         IsolatedFromAbove, GraphRegionNoTerminator,
                         OpAsmOpInterface,
                         DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Circuit struct/component definition";
  let description = [{
      Describes a struct/component in a circuit.

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
  }];

  let hasRegionVerifier = 1;
}

def LLZK_ConstReadOp
    : LLZK_Op<
          "read_const", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Read value from a struct/component constant parameter.";
  let description = [{
    This operation reads the value from the named constant parameter of
    the struct/component in which this op appears. The op itself puts
    some restriction on the type of this value, but leaves it to a later
    type-checking pass to ensure the struct parameters are instantiated
    with types matching the uses of the parameter within the struct.
  }];

  let arguments = (ins FlatSymbolRefAttr:$const_name);
  let results = (outs ConstReadType:$val);

  let assemblyFormat = [{ $const_name attr-dict `:` type($val) }];
}

def LLZK_FieldDefOp
    : LLZK_Op<"field", [HasParent<"::llzk::StructDefOp">,
                        DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                        Symbol]> {
  let summary = "Struct/component field";
  let description = [{
    Describes a field in a struct/component.

    Example:

    ```llzk
    llzk.field @f1 : !llzk.felt
    llzk.field @f2 : !llzk.felt {llzk.pub}
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<AnyLLZKType>:$type);

  let assemblyFormat = [{
    $sym_name `:` $type attr-dict
  }];

  let extraClassDeclaration = [{
    bool hasPublicAttr();
  }];
}

def LLZK_FieldReadOp : LLZK_FieldRefOp<"readf"> {
  let summary = "Read value from a struct/component field.";
  let description = [{
    This operation reads the value from the named field from a struct/component.
  }];

  let arguments = (ins LLZK_StructType:$component,
      FlatSymbolRefAttr:$field_name);
  let results = (outs AnyLLZKType:$val);

  let assemblyFormat = [{
    $component `[` $field_name `]` attr-dict `:` type($component) `,` type($val)
  }];
}

def LLZK_FieldWriteOp : LLZK_FieldRefOp<"writef", [ComputeOnly]> {
  let summary = "Write value to a struct/component field.";
  let description = [{
    This operation writes the value to the named field from a struct/component.
  }];

  let arguments = (ins LLZK_StructType:$component,
      FlatSymbolRefAttr:$field_name, AnyLLZKType:$val);

  let assemblyFormat = [{
    $component `[` $field_name `]` `=` $val attr-dict `:` type($component) `,` type($val)
  }];
}

def LLZK_CreateStructOp
    : LLZK_Op<"new_struct", [ComputeOnly,
                             DeclareOpInterfaceMethods<
                                 OpAsmOpInterface, ["getAsmResultNames"]>,
                             DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
  let summary = "creates a new struct";
  let description = [{
    This operation creates a new, uninitialized instance of a struct.

    Example:

    ```llzk
    %self = llzk.new_struct : !llzk.struct<@Reg>
    ```
  }];

  let results = (outs LLZK_StructType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

//===------------------------------------------------------------------===//
// Array operations
//===------------------------------------------------------------------===//

def LLZK_CreateArrayOp
    : LLZK_Op<
          "new_array",
          [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>,
           DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
           VariadicTypesMatchWith<
               "operand types match result type", "result", "elements",
               "resultTypeToElementsTypes($_self)", "std::equal_to<>()">]> {
  let summary = "creates an array";
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
  }];

  let arguments = (ins Variadic<AnyLLZKType>:$elements);
  let results = (outs LLZK_ArrayType:$result);

  // This uses the custom parseInferredArrayType function to compute the type
  //  of '$elements' to match the type of '$result', except when '$elements'
  //  is empty, then the type of '$elements' must also be empty (size == 0).
  let assemblyFormat = [{
        $elements attr-dict `:` type($result)
        `` custom<InferredArrayType>(type($elements), ref($elements), ref(type($result)))
      }];

  let extraClassDeclaration = [{
    static ::mlir::ParseResult parseInferredArrayType(::mlir::AsmParser &parser,
        ::llvm::SmallVector<::mlir::Type,1> &elementsTypes,
        ::mlir::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> elements,
        ::mlir::Type resultType
    );
    static void printInferredArrayType(::mlir::AsmPrinter &printer, CreateArrayOp,
        ::mlir::Operation::operand_range::type_range, ::mlir::Operation::operand_range, ::mlir::Type
    );
  private:
    static ::llvm::SmallVector<::mlir::Type> resultTypeToElementsTypes(::mlir::Type resultType);
  }];
}

def LLZK_ReadArrayOp
    : LLZK_Op<
          "readarr", [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                      ArrayTypeElemsUnifyWithResultCustomInfer<"arr_ref">]> {
  let summary = "Loads a value from a LLZK array";
  let description = [{
    This operation loads the value from an array at position `$index`.

    Example of 1-dimensional array:
    ```llzk
    %i = index.constant 0
    %0 = llzk.new_array %a, %b, %c : !llzk.array<3 x !llzk.felt>
    // %1 is now equal to %a
    %1 = llzk.readarr %0[%i] : !llzk.array<3 x !llzk.felt>, !llzk.felt
    ```

    Example of 3-dimensional array:
    ```llzk
    %i = index.constant 0
    %j = index.constant 1
    %k = index.constant 4
    %0 = llzk.new_array ... : !llzk.array<3,10,15 x !llzk.felt>
    // %1 is now equal to %a
    %1 = llzk.readarr %0[%i, %j, %k] : !llzk.array<3,10,15 x !llzk.felt>, !llzk.felt
    ```
  }];

  let arguments = (ins LLZK_ArrayType:$arr_ref, Variadic<Index>:$indices);
  let results = (outs AnyLLZKType:$result);

  let assemblyFormat = [{
    $arr_ref `[` $indices `]` attr-dict `:` type($arr_ref) `,` type($result)
  }];
}

def LLZK_WriteArrayOp
    : LLZK_Op<"writearr", [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                           ArrayElemTypeUnifyWith<"arr_ref", "rvalue">]> {
  let summary = "Writes a value into a LLZK array";
  let description = [{
    This operation writes a value into an array at position `$index`.

    Example of 1-dimensional array:

    ```llzk
    %i = index.constant 0
    %0 = llzk.constfelt 42
    %1 = llzk.new_array %a, %b, %c : !llzk.array<3 x !llzk.felt>
    // The array now is [%0, %b, %c]
    llzk.writearr %1[%i] = %0 : !llzk.array<3 x !llzk.felt>, !llzk.felt
    ```

    Example of 2-dimensional array:

    ```llzk
    %i = index.constant 0
    %j = index.constant 0
    %0 = llzk.constfelt 42
    %1 = llzk.new_array %a, %b, %c, %d : !llzk.array<2,2 x !llzk.felt>
    // The array now is [[%0, %b], [%c, %d]]
    llzk.writearr %1[%i, %j] = %0 : !llzk.array<2,2 x !llzk.felt>, !llzk.felt
    ```
  }];

  let arguments = (ins LLZK_ArrayType:$arr_ref, Variadic<Index>:$indices,
      AnyLLZKType:$rvalue);

  let assemblyFormat = [{
    $arr_ref `[` $indices `]` `=` $rvalue attr-dict `:` type($arr_ref) `,` type($rvalue)
  }];
}

def LLZK_ExtractArrayOp
    : LLZK_Op<"extractarr", [Pure, InferTypeOpAdaptorWithIsCompatible]> {
  let summary = "Get subarray from a multi-dimensional array.";
  let description = [{
    Takes an N-dimensional array and K indices and extracts the (N-K)-dimensional
    array by applying the given indices to the first `K` dimensions of the array.
    Error if `K > N`. Use `readarr` instead if `K == N`.

    Extracting a 1-D array from 3-D array by selecting the index of 2 dimensions:
    ```llzk
    %i = index.constant 1
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
    $arr_ref `[` $indices `]` attr-dict `:` type($arr_ref)
  }];
}

def LLZK_ArrayLengthOp : LLZK_Op<"array_len", [Pure]> {
  let summary = "returns the length of an array";
  let description = [{
    This operation returns the size of the specified dimension of an array.

    Example:

    ```llzk
    %a = llzk.new_array : !llzk.array<2,3 x !llzk.felt>
    %0 = index.constant 0
    %x = llzk.array_len %a, %0 : !llzk.array<2,3 x !llzk.felt> // result is 2
    %1 = index.constant 1
    %y = llzk.array_len %a, %1 : !llzk.array<2,3 x !llzk.felt> // result is 3
    ```
  }];

  let arguments = (ins LLZK_ArrayType:$array, Index:$dim);
  let results = (outs Index:$length);

  let assemblyFormat = "$array `,` $dim `:` type($array) attr-dict";
}

//===------------------------------------------------------------------===//
// Constraint emit operations
//===------------------------------------------------------------------===//

class LLZK_EmitOp<string mnemonic, list<Trait> traits = []>
    : LLZK_Op<mnemonic,
              traits#[InStructFunctionNamed<"::llzk::FUNC_NAME_CONSTRAIN">]> {

  let assemblyFormat = [{
    $lhs `,` $rhs `:` type($lhs)
    `` custom<InferredOrParsedType>(type($rhs), ref(type($lhs)))
    attr-dict
  }];

  let extraClassDeclaration = [{
    static ::mlir::Type inferRHS(::mlir::Type lhsType);

    static ::mlir::ParseResult parseInferredOrParsedType(::mlir::AsmParser &parser,
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

    static void printInferredOrParsedType(::mlir::AsmPrinter &printer,
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
  let summary = "";
  let description = [{}];

  let arguments = (ins EmitEqType:$lhs, EmitEqType:$rhs);
}

def LLZK_EmitContainmentOp
    : LLZK_EmitOp<"emit_in", [ArrayElemTypeUnifyWith<"lhs", "rhs">]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins LLZK_ArrayType:$lhs, AnyLLZKType:$rhs);
}

//===------------------------------------------------------------------===//
// Field element operators
//===------------------------------------------------------------------===//

def LLZK_AddFeltOp : LLZK_BinaryOpBase<"add", LLZK_FeltType, [Commutative]> {
  let summary = "Addition operator for field elements";
  let description = [{}];
}

def LLZK_SubFeltOp : LLZK_BinaryOpBase<"sub", LLZK_FeltType> {
  let summary = "Subtraction operator for field elements";
  let description = [{}];
}

def LLZK_MulFeltOp : LLZK_BinaryOpBase<"mul", LLZK_FeltType, [Commutative]> {
  let summary = "Multiplication operator for field elements";
  let description = [{}];
}

def LLZK_DivFeltOp : LLZK_BinaryOpBase<"div", LLZK_FeltType> {
  let summary = "Division operator for field elements";
  let description = [{}];
}

def LLZK_ModFeltOp : LLZK_BinaryOpBase<"mod", LLZK_FeltType> {
  let summary = "Modulus/remainder operator for field elements";
  let description = [{}];
}

def LLZK_NegFeltOp : LLZK_UnaryOpBase<"neg", LLZK_FeltType> {
  let summary = "Negation operator for field elements";
  let description = [{}];
}

def LLZK_InvFeltOp : LLZK_UnaryOpBase<"inv", LLZK_FeltType, [ComputeOnly]> {
  let summary = "Inverse operator for field elements";
  let description = [{}];
}

def LLZK_AndFeltOp
    : LLZK_BinaryOpBase<"bit_and", LLZK_FeltType, [ComputeOnly, Commutative]> {
  let summary = "Bitwise AND operator for field elements";
  let description = [{}];
}

def LLZK_OrFeltOp
    : LLZK_BinaryOpBase<"bit_or", LLZK_FeltType, [ComputeOnly, Commutative]> {
  let summary = "Bitwise OR operator for field elements";
  let description = [{}];
}

def LLZK_XorFeltOp
    : LLZK_BinaryOpBase<"bit_xor", LLZK_FeltType, [ComputeOnly, Commutative]> {
  let summary = "Bitwise XOR operator for field elements";
  let description = [{}];
}

def LLZK_NotFeltOp : LLZK_UnaryOpBase<"bit_not", LLZK_FeltType, [ComputeOnly]> {
  let summary = "Bit flip operator for field elements";
  let description = [{}];
}

def LLZK_ShlFeltOp : LLZK_BinaryOpBase<"shl", LLZK_FeltType, [ComputeOnly]> {
  let summary = "Left shift operator for field elements";
  let description = [{}];
}

def LLZK_ShrFeltOp : LLZK_BinaryOpBase<"shr", LLZK_FeltType, [ComputeOnly]> {
  let summary = "Right shift operator for field elements";
  let description = [{}];
}

// Match format of Index comparisons (for now)
def LLZK_CmpOp : LLZK_Op<"cmp", [Pure]> {
  let summary = "field element comparison operation";
  let description = [{
    The `llzk.cmp` operation takes two field element values and compares them
    according to the comparison predicate and returns an `i1`. The following
    comparisons are supported:

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
  let assemblyFormat = "`` $predicate `(` $lhs `,` $rhs `)` attr-dict";
}

def LLZK_AndBoolOp : LLZK_BinaryOpBase<"and", I1, [ComputeOnly, Commutative]> {
  let summary = "logical AND";
  let description = [{
    Logical AND (i.e. conjunction) operator. The type of operands and return value is `i1` (i.e. boolean).
  }];
}

def LLZK_OrBoolOp : LLZK_BinaryOpBase<"or", I1, [ComputeOnly, Commutative]> {
  let summary = "logical OR";
  let description = [{
    Logical OR (i.e. disjunction) operator. The type of operands and return value is `i1` (i.e. boolean).
  }];
}

def LLZK_XorBoolOp : LLZK_BinaryOpBase<"xor", I1, [ComputeOnly, Commutative]> {
  let summary = "logical XOR";
  let description = [{
    Logical ZOR (i.e. exclusive disjunction) operator. The type of operands and return value is `i1` (i.e. boolean).
  }];
}

def LLZK_NotBoolOp : LLZK_UnaryOpBase<"not", I1, [ComputeOnly]> {
  let summary = "logical NOT";
  let description = [{
    Logical NOT (i.e. negation) operator. The type of operand and return value is `i1` (i.e. boolean).
  }];
}

def LLZK_IntToFeltOp : LLZK_Op<"tofelt", [Pure]> {
  let summary = "converts an integer into a felt";
  let description = [{
    Transforms any builtin integer value into a felt value.

    Example:

    ```llzk
    %0 = llzk.cmp lt(%a, %b)
    %1 = llzk.tofelt %0 : i1
    ```
  }];

  let arguments = (ins AnyLLZKIntType:$value);
  let results = (outs LLZK_FeltType:$result);
  let assemblyFormat = "$value `:` type($value) attr-dict";
}

def LLZK_FeltToIndexOp : LLZK_Op<"toindex", [Pure]> {
  let summary = "converts a felt into an index type";
  let description = [{
    Transforms a felt type into an index to allow being used as array indices.

    Example:
    ```llzk
    %0 = llzk.toindex %a 
    %1 = llzk.readarr %b[%0]
    ```
  }];

  let arguments = (ins LLZK_FeltType:$value);
  let results = (outs Index:$result);
  let assemblyFormat = "$value attr-dict";
}

#endif // LLZK_OPS
