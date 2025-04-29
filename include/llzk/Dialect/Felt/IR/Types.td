//===-- Types.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_FELT_TYPES
#define LLZK_FELT_TYPES

include "llzk/Dialect/Felt/IR/Dialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def LLZK_FeltType : TypeDef<FeltDialect, "Felt"> {
  let mnemonic = "felt";
  let summary = "finite field element";
  let description = [{}];

  let extraClassDeclaration = [{
    static ::llzk::felt::FeltType get(::mlir::MLIRContext *context) {
      return Base::get(context);
    }
  }];
}

#endif // LLZK_FELT_TYPES
