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

//===------------------------------------------------------------------===//
// Struct Operations
//===------------------------------------------------------------------===//

def LLZK_StructDefOp
    : LLZK_Op<"struct", [HasParent<"::mlir::ModuleOp">, Symbol, SymbolTable,
                         IsolatedFromAbove, GraphRegionNoTerminator,
                         OpAsmOpInterface]> {
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

  let arguments = (ins SymbolNameAttr:$sym_name,
      OptionalAttr<FlatSymbolRefArrayAttr>:$const_params);

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name (`<` $const_params^ `>`)? $body attr-dict
  }];

  let extraClassDeclaration = [{
    /// Gets the StructType representing this struct.
    StructType getType();

    /// Gets the FieldDefOp that defines the field in this
    /// structure with the given name, if present.
    FieldDefOp getFieldDef(::mlir::StringAttr fieldName);

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

def LLZK_FieldWriteOp : LLZK_FieldRefOp<"writef"> {
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
    : LLZK_Op<"new_struct", [DeclareOpInterfaceMethods<
                                 OpAsmOpInterface, ["getAsmResultNames"]>,
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
          [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
           Pure,
           TypesMatchWith<
               "operand types match result type", "result", "elements",
               "::llvm::SmallVector<::mlir::Type, 2>("
               "::llvm::cast<::llzk::ArrayType>($_self).getNumElements(), "
               "::llvm::cast<::llzk::ArrayType>($_self).getElementType())">]> {
  let summary = "creates an array";
  let description = [{
    This operation creates a new array with the given elements.
    The arguments are passed as a flat array but get arranged
    according the shape declared in the type.

    Examples:

    ```llzk
    %0 = llzk.new_array %a, %b, %c : !llzk.array<3 x !llzk.felt>
    %1 = llzk.new_array %a, %b, %c, %d : !llzk.array<2x2 x !llzk.felt>
    ```
  }];

  let arguments = (ins Variadic<AnyLLZKType>:$elements);
  let results = (outs LLZK_ArrayType:$result);

  let assemblyFormat = "$elements attr-dict `:` type($result)";
}

def LLZK_ReadArrayOp
    : LLZK_Op<
          "readarr",
          [TypesMatchWith<
              "result types match with array content", "lvalue", "result",
              "::llvm::cast<::llzk::ArrayType>($_self).getElementType()">]> {
  let summary = "Loads a value from a LLZK array";
  let description = [{
    This operation loads the value from an array at position `$index`.

    Exampl̀e of 1-dimensional array:
    ```llzk
    %i = index.constant 0
    %0 = llzk.new_array %a, %b, %c : !llzk.array<3 x !llzk.felt>
    // %1 is now equal to %a
    %1 = llzk.readarr %0[%i] : !llzk.array<3 x !llzk.felt>, !llzk.felt
    ```

    Exampl̀e of 3-dimensional array:
    ```llzk
    %i = index.constant 0
    %j = index.constant 1
    %k = index.constant 4
    %0 = llzk.new_array ... : !llzk.array<3x10x15 x !llzk.felt>
    // %1 is now equal to %a
    %1 = llzk.readarr %0[%i, %j, %k] : !llzk.array<3x10x15 x !llzk.felt>, !llzk.felt
    ```
  }];

  let arguments = (ins LLZK_ArrayType:$lvalue, Variadic<Index>:$indices);
  let results = (outs AnyLLZKType:$result);

  let assemblyFormat = [{
    $lvalue `[` $indices `]` attr-dict `:` type($lvalue) `,` type($result)
  }];
}

def LLZK_WriteArrayOp
    : LLZK_Op<
          "writearr",
          [TypesMatchWith<
              "rvalue types match with array content", "lvalue", "rvalue",
              "::llvm::cast<::llzk::ArrayType>($_self).getElementType()">]> {
  let summary = "Writes a value into a LLZK array";
  let description = [{
    This operation writes a value into an array at position `$index`.

    Exampl̀e of 1-dimensional array:

    ```llzk
    %i = index.constant 0
    %0 = llzk.constfelt 42
    %1 = llzk.new_array %a, %b, %c : !llzk.array<3 x !llzk.felt>
    // The array now is [%0, %b, %c]
    llzk.writearr %1[%i] = %0 : !llzk.array<3 x !llzk.felt>, !llzk.felt
    ```

    Exampl̀e of 2-dimensional array:

    ```llzk
    %i = index.constant 0
    %j = index.constant 0
    %0 = llzk.constfelt 42
    %1 = llzk.new_array %a, %b, %c, %d : !llzk.array<2x2 x !llzk.felt>
    // The array now is [[%0, %b], [%c, %d]]
    llzk.writearr %1[%i, %j] = %0 : !llzk.array<2x2 x !llzk.felt>, !llzk.felt
    ```

  }];

  let arguments = (ins LLZK_ArrayType:$lvalue, Variadic<Index>:$indices,
      AnyLLZKType:$rvalue);

  let assemblyFormat = [{
    $lvalue `[` $indices `]` `=` $rvalue attr-dict `:` type($lvalue) `,` type($rvalue)
  }];
}

//===------------------------------------------------------------------===//
// Constraint emit operations
//===------------------------------------------------------------------===//

def EmitEqType : TypeConstraint<CPred<"::llzk::isValidEmitEqType($_self)">,
                                "non-struct type">;

def LLZK_EmitEqualityOp
    : LLZK_Op<"emit_eq", [InStructFunctionNamed<"::llzk::FUNC_NAME_CONSTRAIN">,
                          Pure, Commutative, SameTypeOperands,
                          ElementwiseMappable,
]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins EmitEqType:$lhs, EmitEqType:$rhs);

  let assemblyFormat = [{
    $lhs `,` $rhs `:` type($rhs) attr-dict
  }];
}

def LLZK_EmitContainmentOp
    : LLZK_Op<"emit_in",
              [InStructFunctionNamed<"::llzk::FUNC_NAME_CONSTRAIN">, Pure,
               TypesMatchWith<"lhs type matches rhs element type", "rhs", "lhs",
                              "::llvm::cast<::llzk::ArrayType>($_self)."
                              "getElementType()">]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins AnyLLZKType:$lhs, LLZK_ArrayType:$rhs);

  let assemblyFormat = [{
    $lhs `,` $rhs `:` type($rhs) attr-dict
  }];
}

//===------------------------------------------------------------------===//
// Field element operators
//===------------------------------------------------------------------===//

def LLZK_AddFeltOp : LLZK_BinaryFeltOp<"add", [Commutative]> {
  let summary = "Addition operator for field elements";
  let description = [{}];
}

def LLZK_SubFeltOp : LLZK_BinaryFeltOp<"sub"> {
  let summary = "Subtraction operator for field elements";
  let description = [{}];
}

def LLZK_MulFeltOp : LLZK_BinaryFeltOp<"mul", [Commutative]> {
  let summary = "Multiplication operator for field elements";
  let description = [{}];
}

def LLZK_DivFeltOp : LLZK_BinaryFeltOp<"div"> {
  let summary = "Division operator for field elements";
  let description = [{}];
}

def LLZK_ModFeltOp : LLZK_BinaryFeltOp<"mod"> {
  let summary = "Modulus/remainder operator for field elements";
  let description = [{}];
}

def LLZK_NegFeltOp : LLZK_UnaryFeltOp<"neg"> {
  let summary = "Negation operator for field elements";
  let description = [{}];
}

def LLZK_InvFeltOp : LLZK_UnaryFeltOp<"inv", [ComputeOnly]> {
  let summary = "Inverse operator for field elements";
  let description = [{}];
}

def LLZK_AndFeltOp : LLZK_BinaryFeltOp<"bit_and", [ComputeOnly, Commutative]> {
  let summary = "Bitwise AND operator for field elements";
  let description = [{}];
}

def LLZK_OrFeltOp : LLZK_BinaryFeltOp<"bit_or", [ComputeOnly, Commutative]> {
  let summary = "Bitwise OR operator for field elements";
  let description = [{}];
}

def LLZK_XorFeltOp : LLZK_BinaryFeltOp<"bit_xor", [ComputeOnly, Commutative]> {
  let summary = "Bitwise XOR operator for field elements";
  let description = [{}];
}

def LLZK_NotFeltOp : LLZK_UnaryFeltOp<"bit_not", [ComputeOnly]> {
  let summary = "Bit flip operator for field elements";
  let description = [{}];
}

def LLZK_ShlFeltOp : LLZK_BinaryFeltOp<"shl", [ComputeOnly]> {
  let summary = "Left shift operator for field elements";
  let description = [{}];
}

def LLZK_ShrFeltOp : LLZK_BinaryFeltOp<"shr", [ComputeOnly]> {
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

def LLZK_AndBoolOp : LLZK_BinaryBoolOp<"and", [ComputeOnly, Commutative]> {
  let summary = "logical AND";
  let description = [{
    Logical AND (i.e. conjunction) operator. The type of operands and return value is `i1` (i.e. boolean).
  }];
}

def LLZK_OrBoolOp : LLZK_BinaryBoolOp<"or", [ComputeOnly, Commutative]> {
  let summary = "logical OR";
  let description = [{
    Logical OR (i.e. disjunction) operator. The type of operands and return value is `i1` (i.e. boolean).
  }];
}

def LLZK_XorBoolOp : LLZK_BinaryBoolOp<"xor", [ComputeOnly, Commutative]> {
  let summary = "logical XOR";
  let description = [{
    Logical ZOR (i.e. exclusive disjunction) operator. The type of operands and return value is `i1` (i.e. boolean).
  }];
}

def LLZK_NotBoolOp : LLZK_UnaryBoolOp<"not", [ComputeOnly]> {
  let summary = "logical NOT";
  let description = [{
    Logical NOT (i.e. negation) operator. The type of operand and return value is `i1` (i.e. boolean).
  }];
}

#endif // LLZK_OPS
