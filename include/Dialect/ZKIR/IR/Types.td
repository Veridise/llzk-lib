#ifndef ZKIR_TYPES
#define ZKIR_TYPES

include "Dialect/ZKIR/IR/Attrs.td"
include "Dialect/ZKIR/IR/Dialect.td"
include "Dialect/ZKIR/IR/Enums.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"

class ZKIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ZKIRDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def ZKIR_FeltType : ZKIR_Type<"Felt", "felt"> {
  let summary = "finite field element";

  let extraClassDeclaration = [{
    static ::zkir::FeltType get(::mlir::MLIRContext *context) {
      return Base::get(context);
    }
  }];
}

def ZKIR_StructType : ZKIR_Type<"Struct", "struct"> {
  let summary = "circuit component type";
  let description = [{}];
  let parameters = (ins "::mlir::SymbolRefAttr":$name);

  let assemblyFormat = [{ `<` $name `>` }];

  let extraClassDeclaration = [{
    /// Gets the `struct` op that defines this struct.
    /// Provided `op` is used as a starting point for the lookup.
    /// Should not be assumed to be non-`null` as we don't verify all
    /// types during verification.
    StructDefOp getDefinition(::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op);
  
    // Verifies that this type references a valid struct, relative to the given
    // `op`.
    mlir::LogicalResult verifySymbol(
        mlir::SymbolTableCollection &symbolTable, mlir::Operation *op);
  }];
}

def ZKIR_ArrayType : ZKIR_Type<"Array", "array", []> {
  let summary = "array type with a static size and homogeneous element type";
  let description = [{}];

  //TODO: I want to restrict element type to be AnyZKIRType but that construct is not
  //  generated in the *.inc files and it's circular so I can't use before it's defined.
  let parameters = (ins "::mlir::Type":$elementType, "uint64_t":$numElements);
  let assemblyFormat = [{
    `<` $numElements `x` $elementType `>`
  }];
}

def AnyZKIRType : AnyTypeOf<[Index, ZKIR_FeltType, ZKIR_StructType, ZKIR_ArrayType]>;

#endif // ZKIR_TYPES
