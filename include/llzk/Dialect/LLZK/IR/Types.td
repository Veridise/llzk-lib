#ifndef LLZK_TYPES
#define LLZK_TYPES

include "llzk/Dialect/LLZK/IR/Dialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"

class LLZK_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<LLZKDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def AnyLLZKType : Type<CPred<"::llzk::isValidType($_self)">,
                       "a valid LLZK type", "::mlir::Type">;

def LLZK_FeltType : LLZK_Type<"Felt", "felt"> {
  let summary = "finite field element";
  let description = [{}];

  let extraClassDeclaration = [{
    static ::llzk::FeltType get(::mlir::MLIRContext *context) {
      return Base::get(context);
    }
  }];
}

def LLZK_StructType : LLZK_Type<"Struct", "struct"> {
  let summary = "circuit component";
  let description = [{}];

  let parameters =
      (ins TypeParameter<
           "::mlir::SymbolRefAttr",
           "Fully-qualified name of the struct definition.">:$nameRef,
          OptionalParameter<"::mlir::ArrayAttr", "Struct parameters">:$params);

  let assemblyFormat = [{ `<` $nameRef ( `<` $params^ `>` )? `>` }];

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Gets the `struct` op that defines this struct. Provided `op` is
    /// used as a starting point for the lookup. Should not be assumed to
    /// be non-`null` as we don't verify all types during verification.
    ::mlir::FailureOr<SymbolLookupResult<StructDefOp>> getDefinition
        (::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op);

    // Verifies that this type references a valid struct, relative to the given `op`.
    ::mlir::LogicalResult verifySymbolRef(::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op);
  }];
}

def LLZK_ArrayType : LLZK_Type<"Array", "array", [ShapedTypeInterface]> {
  let summary = "n-dimensional array";
  let description = [{
    Array type with a ranked shape and homogeneous element type.
    It can only be instantiated with the following types:
      - Any LLZK type
      - IndexType
      - Unsigned integers of 1 bit (aka booleans)

    ```llzk
    // Example array of 5 by 2 elements of `Felt` type.
    !llzk.array<5,2 x !llzk.felt>

    // Example array using a struct parameter for one dimension.
    !llzk.array<5,@A x index>
    ```
  }];

  let parameters =
      (ins TypeParameter<"::mlir::Type",
                         "Type of all elements within the array.">:$elementType,
          ArrayRefParameter<
              "::mlir::Attribute",
              "List of array dimension size specifiers.">:$dimensionSizes,
          ArrayRefParameter<
              "int64_t",
              "Array shape, for ShapedTypeInterface, computed from  .">:$shape);

  let assemblyFormat = [{ 
    `<` custom<AttrVec>($dimensionSizes) `x` $elementType `>`
        custom<DerivedShape>($shape, ref($dimensionSizes))
  }];

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Returns if this type is ranked, i.e. it has a known number of dimensions.
    /// LLZK arrays are always ranked, i.e. the number of dimensions is known.
    /// Required by the ShapedTypeInterface interface.
    inline bool hasRank() const { return true; }

    /// Clone this type with the given shape and element type. If the
    /// provided shape is `std::nullopt`, the current shape of the type is used.
    /// Required by the ShapedTypeInterface interface.
    ArrayType cloneWith(std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType) const;

    /// Returns the number of elements in the array.
    /// Aborts if the array has a dynamic dimension.
    /// Required by the ShapedTypeInterface interface.
    int64_t getNumElements() const;
  }];
}

#endif // LLZK_TYPES
