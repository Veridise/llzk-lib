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
include "llzk/Dialect/Felt/IR/Types.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"

class LLZK_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<LLZKDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def AnyLLZKType : Type<CPred<"::llzk::isValidType($_self)">,
                       "a valid LLZK type", "::mlir::Type">;

def GlobalDefType : Type<CPred<"::llzk::isValidGlobalType($_self)">,
                         "any LLZK type except non-constant types">;

def EmitEqType
    : Type<CPred<"::llzk::isValidEmitEqType($_self)">,
           "any LLZK type, excluding non-Signal struct and string types">;

def ConstReadType : Type<CPred<"::llzk::isValidConstReadType($_self)">,
                         "integral, felt, or typevar type">;

def AnyLLZKIntType : AnyTypeOf<[I1, Index]>;

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
      field @x : !array.array<2 x !llzk.tvar<@Ty>>
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

#endif // LLZK_TYPES
