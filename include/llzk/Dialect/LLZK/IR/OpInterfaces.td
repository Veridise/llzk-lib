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

#endif // LLZK_OP_INTERFACES
