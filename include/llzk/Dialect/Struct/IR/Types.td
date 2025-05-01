//===-- Types.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_STRUCT_TYPES
#define LLZK_STRUCT_TYPES

include "llzk/Dialect/Shared/Types.td"
include "llzk/Dialect/Struct/IR/Dialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class StructDialectType<string name, string typeMnemonic,
                        list<Trait> traits = []>
    : TypeDef<StructDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def LLZK_StructType : StructDialectType<"Struct", "type"> {
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

#endif // LLZK_STRUCT_TYPES
