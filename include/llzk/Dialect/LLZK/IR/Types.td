//===-- Types.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

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

def GlobalDefType : Type<CPred<"::llzk::isValidGlobalType($_self)">,
                         "any LLZK type except non-constant types">;

def EmitEqType : Type<CPred<"::llzk::isValidEmitEqType($_self)">,
                      "any LLZK type, excluding non-Signal struct and string types">;

def ConstReadType : Type<CPred<"::llzk::isValidConstReadType($_self)">,
                         "integral, felt, or typevar type">;

def ArrayElemType : Type<CPred<"::llzk::isValidArrayElemType($_self)">,
                         "a valid array element type">;

def AnyLLZKIntType : AnyTypeOf<[I1, Index]>;

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

  let assemblyFormat =
      [{ `<` $nameRef ( `<` custom<StructParams>($params)^ `>` )? `>` }];

  let genVerifyDecl = 1;

  let skipDefaultBuilders = 1;
  let builders = [TypeBuilderWithInferredContext<
                      (ins "::mlir::SymbolRefAttr":$structName), [{
                    return $_get(structName.getContext(), structName, ::mlir::ArrayAttr());
                  }]>,
                  TypeBuilderWithInferredContext<
                      (ins "::mlir::SymbolRefAttr":$structName,
                          "::mlir::ArrayAttr":$params),
                      [{
                    ::mlir::MLIRContext *ctx = structName.getContext();
                    if (params) {
                      params = ::mlir::ArrayAttr::get(ctx, forceIntAttrTypes(params.getValue()));
                    }
                    return $_get(ctx, structName, params);
                  }]>,
                  TypeBuilderWithInferredContext<
                      (ins "::mlir::SymbolRefAttr":$structName,
                          "::llvm::ArrayRef<::mlir::Attribute>":$paramsRef),
                      [{
                    ::mlir::MLIRContext *ctx = structName.getContext();
                    ::llvm::SmallVector<::mlir::Attribute> p = forceIntAttrTypes(paramsRef);
                    return $_get(ctx, structName, ::mlir::ArrayAttr::get(ctx, p));
                  }]>];

  let extraClassDeclaration = [{
    /// Gets the `struct` op that defines this struct. Provided `op` is
    /// used as a starting point for the lookup. Should not be assumed to
    /// be non-`null` as we don't verify all types during verification.
    ::mlir::FailureOr<SymbolLookupResult<StructDefOp>> getDefinition
        (::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op) const;

    // Verifies that this type references a valid struct, relative to the given `op`.
    ::mlir::LogicalResult verifySymbolRef(::mlir::SymbolTableCollection &symbolTable, ::mlir::Operation *op);

    /// Returns wether the struct this type refers to has fields marked as columns.
    /// A lookup is necessary first and will forward the failure state if it fails.
    mlir::LogicalResult hasColumns(mlir::SymbolTableCollection &symbolTable, mlir::Operation *op) const;
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
          ArrayRefParameter<"int64_t",
                            "Array shape, for ShapedTypeInterface, computed "
                            "from `$dimensionSizes`.">:$shape);

  // The custom<DerivedShape>() section is 0-length (it is used to compute the
  // "$shape" attribute from the "$dimensionSizes" attribute) and the `` before
  // it avoids excess space in the format. Additionally, it is placed right
  // after parsing "$dimensionSizes" so the source location pointer in the
  // parser remains at "$dimensionSizes" in case an error occurs during the
  // conversion.
  let assemblyFormat = [{
    `<` custom<AttrVec>($dimensionSizes)
    `` custom<DerivedShape>($shape, ref($dimensionSizes))
    `x` $elementType `>`
  }];

  let genVerifyDecl = 1;

  let skipDefaultBuilders = 1;
  let builders =
      [TypeBuilderWithInferredContext<
           (ins "::mlir::Type":$elementType,
               "::llvm::ArrayRef<::mlir::Attribute>":$dimensionSizes),
           [{
      assert(elementType && "element type cannot be null");
      ::llvm::SmallVector<::mlir::Attribute> dimSizes = forceIntAttrTypes(dimensionSizes);
      ::mlir::MLIRContext *ctx = elementType.getContext();
      ::llvm::SmallVector<int64_t> shape;
      ::mlir::LogicalResult res = computeShapeFromDims(emitError, ctx, dimSizes, shape);
      if(::mlir::failed(res)) { return ArrayType(); }
      return $_get(ctx, elementType, dimSizes, shape);
    }]>,
       TypeBuilderWithInferredContext<(ins "::mlir::Type":$elementType,
                                          "::llvm::ArrayRef<int64_t>":$shape),
                                      [{
      assert(elementType && "element type cannot be null");
      ::mlir::MLIRContext *ctx = elementType.getContext();
      ::llvm::SmallVector<::mlir::Attribute> dimSizes;
      ::mlir::LogicalResult res = computeDimsFromShape(ctx, shape, dimSizes);
      if(::mlir::failed(res)) { return ArrayType(); }
      return $_get(ctx, elementType, dimSizes, shape);
    }]>];

  let extraClassDeclaration = [{
    /// Returns if this type is ranked, i.e. it has a known number of dimensions.
    /// LLZK arrays are always ranked, i.e. the number of dimensions is known.
    /// Required by the ShapedTypeInterface interface.
    inline bool hasRank() const { return true; }

    /// Clone this type with the given shape and element type. If the
    /// provided shape is `std::nullopt`, the current shape of the type is used.
    /// Required by the ShapedTypeInterface interface.
    ArrayType cloneWith(std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType) const;

    /// Clone this type with the given dimensions and element type. If the provided
    /// dimensions are `std::nullopt`, the current dimensions of the type are used.
    /// Note: This is preferred over cloneWith(..int64_t..) because this Attribute
    /// version can carry more information than the `int64_t` which must default to
    /// `kDynamic` when the Attribute is anything other than an integer constant.
    ArrayType cloneWith(::mlir::Type elementType, std::optional<::llvm::ArrayRef<::mlir::Attribute>> dimensions = std::nullopt) const;
  }];

  let extraClassDefinition = [{
    namespace {
      /// This definition of `emitError` is used by the `get()` functions generated by the
      /// custom builders for this type. The `getChecked()` functions generated by those same
      /// builders have a parameter with this same name that shadows this definition so the
      /// getChecked() versions will use the function supplied via the parameter. Regardless,
      /// computeShapeFromDims() checks for `nullptr` and generates a default if necessary.
      /// This approach, although a bit hacky, allows a legitimate error function to be used
      /// whenever available, only reverting to a default as needed by the `get()` function.
      const ::llzk::EmitErrorFn emitError = nullptr;
    }
  }];
}

def TypeVarSummary { string r = "type variable"; }

def LLZK_TypeVarType : LLZK_Type<"TypeVar", "tvar"> {
  let summary = TypeVarSummary.r;
  let description = [{
    This type serves as a placeholder for a type that is instantiated
    via a parameter of the struct.

    For example, we can define a struct that holds a size-2 array where the
    type of the elements in the array is specified by a parameter of the
    struct and instantiated with a specific type at the uses of the struct.


    ```llzk
    llzk.struct @A<[@Ty]> {
      field @x : !llzk.array<2 x !llzk.tvar<@Ty>>
      ...
    }
    ```
  }];

  let parameters =
      (ins TypeParameter<"::mlir::FlatSymbolRefAttr",
                         "reference to the struct parameter">:$nameRef);

  let assemblyFormat = [{ `<` $nameRef `>` }];

  let builders = [TypeBuilderWithInferredContext<
      (ins "::mlir::FlatSymbolRefAttr":$nameRef),
      [{ return $_get(nameRef.getContext(), nameRef); }]>];

  let extraClassDeclaration = [{
    inline ::mlir::StringRef getRefName() const { return getNameRef().getValue(); }
  }];
}

// Type constraint matching only the given Type or TypeVarType.
class VarTypeOr<Type allowedConcreteType, string text = TypeVarSummary.r>
    : Type<CPred<"::llvm::isa<llzk::TypeVarType,"#allowedConcreteType
                     .cppClassName#">($_self)">,
           allowedConcreteType.summary#" or "#text, "::mlir::Type">;

def LLZK_String : LLZK_Type<"String", "string"> { let summary = "string type"; }

#endif // LLZK_TYPES
