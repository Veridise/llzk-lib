#ifndef ZKIR_TYPES
#define ZKIR_TYPES

include "zkir/Dialect/ZKIR/IR/Attrs.td"
include "zkir/Dialect/ZKIR/IR/Dialect.td"
include "zkir/Dialect/ZKIR/IR/Enums.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"

class ZKIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ZKIRDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def AnyZKIRType : Type<CPred<"::zkir::isValidZkirType($_self)">,
                       "a valid ZKIR type", "::mlir::Type">;

def ZKIR_FeltType : ZKIR_Type<"Felt", "felt"> {
  let summary = "finite field element";
  let description = [{}];

  let extraClassDeclaration = [{
    static ::zkir::FeltType get(::mlir::MLIRContext *context) {
      return Base::get(context);
    }
  }];
}

def ZKIR_StructType : ZKIR_Type<"Struct", "struct"> {
  let summary = "circuit component";
  let description = [{}];

  let parameters = (ins "::mlir::SymbolRefAttr":$name);
  let assemblyFormat = [{ `<` $name `>` }];

  let extraClassDeclaration = [{
    /// Gets the `struct` op that defines this struct. Provided `op` is
    /// used as a starting point for the lookup. Should not be assumed to
    /// be non-`null` as we don't verify all types during verification.
    ::mlir::FailureOr<StructDefOp> getDefinition(::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op);

    // Verifies that this type references a valid struct, relative to the given `op`.
    ::mlir::LogicalResult verifySymbol(::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op);
  }];
}

def ZKIR_ArrayType : ZKIR_Type<"Array", "array", [ShapedTypeInterface]> {
  let summary = "n-dimensional array";
  let description = [{
    Array type with a ranked shape and homogeneous element type.
    It can only be instantiated with the following types:
      - Any ZKIR type
      - IndexType
      - Unsigned integers of 1 bit (aka booleans)

    Example array of 5 by 2 elements of a `Felt`.
    ```zkir
    !zkir.array<5x2 x !zkir.felt>
    ```
  }];

  let parameters = (ins "::mlir::Type":$elementType,
      ArrayRefParameter<"int64_t">:$shape);
  // Necessary for parsing the dimension list in `$shape`
  let hasCustomAssemblyFormat = true;

  let extraClassDeclaration = [{
    /// Returns if this type is ranked, i.e. it has a known number of dimensions.
    /// Required by the ShapedTypeInterface interface.
    bool hasRank() const;

    /// Clone this type with the given shape and element type. If the
    /// provided shape is `std::nullopt`, the current shape of the type is used.
    /// Required by the ShapedTypeInterface interface.
    ArrayType cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape, Type elementType) const;

    /// Returns the number of elements in the array.
    /// Aborts if the array has a dynamic dimension.
    /// Required by the ShapedTypeInterface interface.
    int64_t getNumElements() const;
  }];

  let genVerifyDecl = 1;
}

#endif // ZKIR_TYPES
