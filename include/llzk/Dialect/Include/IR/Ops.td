//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_INCLUDE_OPS
#define LLZK_INCLUDE_OPS

include "llzk/Dialect/Include/IR/Dialect.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

def LLZK_IncludeOp
    : Op<IncludeDialect, "from", [HasParent<"::mlir::ModuleOp">, Symbol]> {
  let summary = "include operation";
  let description = [{
    This operation imports the contents of another source file in place of itself.

    Example:

    ```llzk
    llzk_include.from "lib.llzk" as @aliasName
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

#endif // LLZK_INCLUDE_OPS
