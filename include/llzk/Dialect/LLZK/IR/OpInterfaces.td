//===-- OpInterfaces.td ------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_OP_INTERFACES
#define LLZK_OP_INTERFACES

include "llzk/Dialect/LLZK/IR/Types.td"

def GlobalRefOpInterface : OpInterface<"GlobalRefOpInterface"> {
  let description = [{
    Common interface for operations that read or write from a global.
  }];
  let cppNamespace = "::llzk";

  let methods = [
      // Requires implementors to have a '$val' argument.
      InterfaceMethod<
          [{Gets the SSA Value that holds the read/write data for the GlobalRefOp.}],
          "::mlir::Value", "getVal", (ins)>,
      // Requires implementors to have a '$name_ref' argument.
      InterfaceMethod<[{Gets the global name attribute from the GlobalRefOp.}],
                      "::mlir::SymbolRefAttr", "getNameRef", (ins)>,
  ];

  let extraClassDeclaration = [{
    /// Gets the definition for the `global` referenced in this op.
    ::mlir::FailureOr<SymbolLookupResult<GlobalDefOp>> getGlobalDefOp(::mlir::SymbolTableCollection &tables);
  }];
}

def FieldRefOpInterface : OpInterface<"FieldRefOpInterface"> {
  let description = [{
    Common interface for operations that read or write from a field.
  }];
  let cppNamespace = "::llzk";

  let methods = [
      // Requires implementors to have a '$val' argument.
      InterfaceMethod<
          [{Gets the SSA Value that holds the read/write data for the FieldRefOp.}],
          "::mlir::Value", "getVal", (ins)>,
      // Requires implementors to have a '$component' argument.
      InterfaceMethod<
          [{Gets the SSA value with the target component from the FieldRefOp.}],
          "::mlir::Value", "getComponent", (ins)>,
      // Requires implementors to have a '$field_name' argument.
      InterfaceMethod<[{Gets the field name attribute from the FieldRefOp.}],
                      "::llvm::StringRef", "getFieldName", (ins)>,
      // Requires the '$component' argument type to be LLZK_StructType.
      InterfaceMethod<[{Gets the struct type of the target component.}],
                      "::llzk::StructType", "getStructType", (ins),
                      /*methodBody=*/[{
        mlir::Type t = $_op.getComponent().getType();
        assert(t.isa<StructType>());
        return t.template cast<StructType>();
      }]>,
  ];

  let extraClassDeclaration = [{
    /// Gets the definition for the `field` referenced in this op.
    ::mlir::FailureOr<SymbolLookupResult<FieldDefOp>> getFieldDefOp(::mlir::SymbolTableCollection &tables);
  }];
}

#endif // LLZK_OP_INTERFACES
