#ifndef ZKIR_OPS
#define ZKIR_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "zkir/Dialect/ZKIR/IR/Attrs.td"
include "zkir/Dialect/ZKIR/IR/Types.td"
include "zkir/Dialect/ZKIR/IR/OpsBase.td"
include "zkir/Dialect/ZKIR/IR/FuncOps.td"

//===------------------------------------------------------------------===//
// Basic constructs
//===------------------------------------------------------------------===//

def ZKIR_FeltConstantOp
    : ZKIR_Op<"constfelt", [ConstantLike, Pure,
                            DeclareOpInterfaceMethods<
                                OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "field element constant";
  let description = [{
    The `zkir.constfelt` operation produces a felt-typed SSA value equal to
    some felt-typed integer constant.

    Example:

    ```zkir
    %0 = zkir.constfelt 42
    ```
  }];

  let arguments = (ins ZKIR_FeltConstAttr:$value);
  let results = (outs ZKIR_FeltType:$result);
  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
}

def ZKIR_FeltNonDetOp
    : ZKIR_Op<"nondetfelt", [ConstantLike, Pure,
                             DeclareOpInterfaceMethods<
                                 OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "uninitialized field element";
  let description = [{
    The `zkir.nondetfelt` operation produces a felt-typed SSA value. This
    can be used in `constrain()` functions in place of expressions that
    cannot be included in constraints.

    Example:

    ```zkir
    %0 = zkir.nondetfelt
    ```
  }];

  let results = (outs ZKIR_FeltType:$result);
  let assemblyFormat = "attr-dict";
}

def ZKIR_IncludeOp
    : ZKIR_Op<"include", [HasParent<"::mlir::ModuleOp">, Symbol]> {
  let summary = "Include operation";
  let description = [{
    The `include` operation allows to import the contents of another source file.

    Example:

    ```zkir
    zkir.include "lib.zkir" as @aliasName
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
      Arg<StrAttr, "source file to include">:$path);

  let assemblyFormat = [{ $path `as` $sym_name attr-dict }];

  let extraClassDeclaration = [{
    static IncludeOp create(::mlir::Location, ::llvm::StringRef name, ::llvm::StringRef path);
    static IncludeOp create(::mlir::Location, ::mlir::StringAttr name, ::mlir::StringAttr path);
    ::mlir::FailureOr<ImportedModuleOp> loadModule();
  }];
}

def ZKIR_ImportedModuleOp : ZKIR_Op<"imported"> {
  // let arguments = (ins "::mlir::OwningOpRef<::mlir::ModuleOp>":$mod);
  let regions = (region AnyRegion:$content);

  let builders = [
      // This builder `std::move`s the smart pointer
      // And assigns the `$content` region to the region pointer of `$mod`
      OpBuilder<(ins "::mlir::OwningOpRef<::mlir::ModuleOp>&&":$mod)>];

  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    static ImportedModuleOp create(::mlir::Location, ::mlir::OwningOpRef<::mlir::ModuleOp>&&);
    static ::mlir::ModuleOp takeModule(ImportedModuleOp&&);
    ::mlir::ModuleOp getModule();
  }];
}

//===------------------------------------------------------------------===//
// Struct Operations
//===------------------------------------------------------------------===//

def ZKIR_StructDefOp
    : ZKIR_Op<"struct", [HasParent<"::mlir::ModuleOp">, Symbol, SymbolTable,
                         IsolatedFromAbove, GraphRegionNoTerminator,
                         OpAsmOpInterface]> {
  let summary = "Circuit struct/component definition";
  let description = [{
      Describes a struct/component in a circuit.

      Example:

      ```zkir
      zkir.struct @ComponentA {
        field @f1 : !zkir.array<5 x index>
        field @f2 : !zkir.felt {zkir.pub}

        func @compute(%p: !zkir.felt) -> !zkir.struct<@ComponentA> {
          %self = new_struct : !zkir.struct<@ComponentA>
          // initialize all fields of `%self` here
          return %self : !zkir.struct<@ComponentA>
        }

        func @constrain(%self: !zkir.struct<@ComponentA>, %p: !zkir.felt) {
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
    /// Gets the FieldDefOp that defines the field in this
    /// structure with the given name, if present.
    FieldDefOp getFieldDef(::mlir::StringAttr fieldName);

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static ::llvm::StringRef getDefaultDialect() {
      return "zkir";
    }
  }];

  let hasRegionVerifier = 1;
}

def ZKIR_FieldDefOp
    : ZKIR_Op<"field", [HasParent<"::zkir::StructDefOp">,
                        DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                        Symbol]> {
  let summary = "Struct/component field";
  let description = [{
    Describes a field in a struct/component.

    Example:

    ```zkir
    zkir.field @f1 : !zkir.felt
    zkir.field @f2 : !zkir.felt {zkir.pub}
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<AnyZKIRType>:$type);

  let assemblyFormat = [{
    $sym_name `:` $type attr-dict
  }];

  let extraClassDeclaration = [{
    bool hasPublicAttr();
  }];
}

def ZKIR_FieldReadOp : ZKIR_FieldRefOp<"readf"> {
  let summary = "Read value from a struct/component field.";
  let description = [{
    This operation reads the value from the named field from a struct/component.
  }];

  let arguments = (ins ZKIR_StructType:$component,
      FlatSymbolRefAttr:$field_name);
  let results = (outs AnyZKIRType:$val);

  let assemblyFormat = [{
    $component `[` $field_name `]` attr-dict `:` type($component) `,` type($val)
  }];
}

def ZKIR_FieldWriteOp : ZKIR_FieldRefOp<"writef"> {
  let summary = "Write value to a struct/component field.";
  let description = [{
    This operation writes the value to the named field from a struct/component.
  }];

  let arguments = (ins ZKIR_StructType:$component,
      FlatSymbolRefAttr:$field_name, AnyZKIRType:$val);

  let assemblyFormat = [{
    $component `[` $field_name `]` `=` $val attr-dict `:` type($component) `,` type($val)
  }];
}

def ZKIR_CreateStructOp
    : ZKIR_Op<"new_struct", [DeclareOpInterfaceMethods<
                                 OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = "creates a new struct";
  let description = [{
    This operation creates a new, uninitialized instance of a struct.

    Example:

    ```zkir
    %self = zkir.new_struct : !zkir.struct<@Reg>
    ```
  }];

  let results = (outs ZKIR_StructType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

//===------------------------------------------------------------------===//
// Array operations
//===------------------------------------------------------------------===//

def ZKIR_CreateArrayOp
    : ZKIR_Op<
          "new_array",
          [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
           Pure,
           TypesMatchWith<
               "operand types match result type", "result", "elements",
               "::llvm::SmallVector<::mlir::Type, 2>("
               "::llvm::cast<::zkir::ArrayType>($_self).getNumElements(), "
               "::llvm::cast<::zkir::ArrayType>($_self).getElementType())">]> {
  let summary = "creates an array";
  let description = [{
    This operation creates a new array with the given elements.
    The arguments are passed as a flat array but get arranged
    according the shape declared in the type.

    Examples:

    ```zkir
    %0 = zkir.new_array %a, %b, %c : !zkir.array<3 x !zkir.felt>
    %1 = zkir.new_array %a, %b, %c, %d : !zkir.array<2x2 x !zkir.felt>
    ```
  }];

  let arguments = (ins Variadic<AnyZKIRType>:$elements);
  let results = (outs ZKIR_ArrayType:$result);

  let assemblyFormat = "$elements attr-dict `:` type($result)";
}

def ZKIR_ReadArrayOp
    : ZKIR_Op<
          "readarr",
          [TypesMatchWith<
              "result types match with array content", "lvalue", "result",
              "::llvm::cast<::zkir::ArrayType>($_self).getElementType()">]> {
  let summary = "Loads a value from a ZKIR array";
  let description = [{
    This operation loads the value from an array at position `$index`.

    Exampl̀e of 1-dimensional array:
    ```zkir
    %i = index.constant 0
    %0 = zkir.new_array %a, %b, %c : !zkir.array<3 x !zkir.felt>
    // %1 is now equal to %a
    %1 = zkir.readarr %0[%i] : !zkir.array<3 x !zkir.felt>, !zkir.felt
    ```

    Exampl̀e of 3-dimensional array:
    ```zkir
    %i = index.constant 0
    %j = index.constant 1
    %k = index.constant 4
    %0 = zkir.new_array ... : !zkir.array<3x10x15 x !zkir.felt>
    // %1 is now equal to %a
    %1 = zkir.readarr %0[%i, %j, %k] : !zkir.array<3x10x15 x !zkir.felt>, !zkir.felt
    ```
  }];

  let arguments = (ins ZKIR_ArrayType:$lvalue, Variadic<Index>:$indices);
  let results = (outs AnyZKIRType:$result);

  let assemblyFormat = [{
    $lvalue `[` $indices `]` attr-dict `:` type($lvalue) `,` type($result)
  }];
}

def ZKIR_WriteArrayOp
    : ZKIR_Op<
          "writearr",
          [TypesMatchWith<
              "rvalue types match with array content", "lvalue", "rvalue",
              "::llvm::cast<::zkir::ArrayType>($_self).getElementType()">]> {
  let summary = "Writes a value into a ZKIR array";
  let description = [{
    This operation writes a value into an array at position `$index`.

    Exampl̀e of 1-dimensional array:

    ```zkir
    %i = index.constant 0
    %0 = zkir.constfelt 42
    %1 = zkir.new_array %a, %b, %c : !zkir.array<3 x !zkir.felt>
    // The array now is [%0, %b, %c]
    zkir.writearr %1[%i] = %0 : !zkir.array<3 x !zkir.felt>, !zkir.felt
    ```

    Exampl̀e of 2-dimensional array:

    ```zkir
    %i = index.constant 0
    %j = index.constant 0
    %0 = zkir.constfelt 42
    %1 = zkir.new_array %a, %b, %c, %d : !zkir.array<2x2 x !zkir.felt>
    // The array now is [[%0, %b], [%c, %d]]
    zkir.writearr %1[%i, %j] = %0 : !zkir.array<2x2 x !zkir.felt>, !zkir.felt
    ```

  }];

  let arguments = (ins ZKIR_ArrayType:$lvalue, Variadic<Index>:$indices,
      AnyZKIRType:$rvalue);

  let assemblyFormat = [{
    $lvalue `[` $indices `]` `=` $rvalue attr-dict `:` type($lvalue) `,` type($rvalue)
  }];
}

//===------------------------------------------------------------------===//
// Constraint emit operations
//===------------------------------------------------------------------===//

def EmitEqType : TypeConstraint<CPred<"::zkir::isValidEmitEqType($_self)">,
                                "non-struct type">;

def ZKIR_EmitEqualityOp
    : ZKIR_Op<"emit_eq", [InStructFunctionNamed<"::zkir::FUNC_NAME_CONSTRAIN">,
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

def ZKIR_EmitContainmentOp
    : ZKIR_Op<"emit_in",
              [InStructFunctionNamed<"::zkir::FUNC_NAME_CONSTRAIN">, Pure,
               TypesMatchWith<"lhs type matches rhs element type", "rhs", "lhs",
                              "::llvm::cast<::zkir::ArrayType>($_self)."
                              "getElementType()">]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins AnyZKIRType:$lhs, ZKIR_ArrayType:$rhs);

  let assemblyFormat = [{
    $lhs `,` $rhs `:` type($rhs) attr-dict
  }];
}

//===------------------------------------------------------------------===//
// Field element operators
//===------------------------------------------------------------------===//

def ZKIR_AddFeltOp : ZKIR_BinaryFeltOp<"add", [Commutative]> {
  let summary = "Addition operator for field elements";
  let description = [{}];
}

def ZKIR_SubFeltOp : ZKIR_BinaryFeltOp<"sub"> {
  let summary = "Subtraction operator for field elements";
  let description = [{}];
}

def ZKIR_MulFeltOp : ZKIR_BinaryFeltOp<"mul", [Commutative]> {
  let summary = "Multiplication operator for field elements";
  let description = [{}];
}

def ZKIR_DivFeltOp : ZKIR_BinaryFeltOp<"div"> {
  let summary = "Division operator for field elements";
  let description = [{}];
}

def ZKIR_ModFeltOp : ZKIR_BinaryFeltOp<"mod"> {
  let summary = "Modulus/remainder operator for field elements";
  let description = [{}];
}

def ZKIR_NegFeltOp : ZKIR_UnaryFeltOp<"neg"> {
  let summary = "Negation operator for field elements";
  let description = [{}];
}

def ZKIR_InvFeltOp : ZKIR_UnaryFeltOp<"inv", [ComputeOnly]> {
  let summary = "Inverse operator for field elements";
  let description = [{}];
}

def ZKIR_AndFeltOp : ZKIR_BinaryFeltOp<"and", [ComputeOnly, Commutative]> {
  let summary = "Bitwise AND operator for field elements";
  let description = [{}];
}

def ZKIR_OrFeltOp : ZKIR_BinaryFeltOp<"or", [ComputeOnly, Commutative]> {
  let summary = "Bitwise OR operator for field elements";
  let description = [{}];
}

def ZKIR_XorFeltOp : ZKIR_BinaryFeltOp<"xor", [ComputeOnly, Commutative]> {
  let summary = "Bitwise XOR operator for field elements";
  let description = [{}];
}

def ZKIR_NotFeltOp : ZKIR_UnaryFeltOp<"not", [ComputeOnly]> {
  let summary = "Bit flip operator for field elements";
  let description = [{}];
}

def ZKIR_ShlFeltOp : ZKIR_BinaryFeltOp<"shl", [ComputeOnly]> {
  let summary = "Left shift operator for field elements";
  let description = [{}];
}

def ZKIR_ShrFeltOp : ZKIR_BinaryFeltOp<"shr", [ComputeOnly]> {
  let summary = "Right shift operator for field elements";
  let description = [{}];
}

// Match format of Index comparisons (for now)
def ZKIR_CmpOp : ZKIR_Op<"cmp", [Pure]> {
  let summary = "field element comparison operation";
  let description = [{
    The `zkir.cmp` operation takes two field element values and compares them
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

    ```zkir
    // Less than comparison.
    %0 = zkir.cmp<lt>(%a, %b)

    // Greater than or equal comparison.
    %1 = zkir.cmp<ge>(%a, %b)

    // Not equal comparison.
    %2 = zkir.cmp<ne>(%a, %b)
    ```
  }];

  let arguments = (ins ZKIR_CmpPredicateAttr:$pred, ZKIR_FeltType:$lhs,
      ZKIR_FeltType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "`` $pred `(` $lhs `,` $rhs `)` attr-dict";
}

#endif // ZKIR_OPS
