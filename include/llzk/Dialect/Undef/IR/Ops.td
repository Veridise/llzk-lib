//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_UNDEF_OPS
#define LLZK_UNDEF_OPS

include "llzk/Dialect/Shared/Types.td"
include "llzk/Dialect/Undef/IR/Dialect.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LLZK_UndefOp : Op<UndefDialect, "undef", [Pure]> {
  let summary = "Creates an undefined value of the specified type.";
  let description = [{
    This operation has no operands or attributes. It represents an undefined
    value of the specified LLZK IR dialect type. This op can be introduced by
    the `llzk-array-to-scalar` pass if there is a read from an array index
    that was not dominated by an earlier write to that same index.

    Example:

    ```llzk
    %0 = undef.undef : !felt.type
    ```
  }];
  let results = (outs AnyLLZKType:$res);
  let assemblyFormat = [{
    `:` type($res) attr-dict
  }];
}

#endif // LLZK_UNDEF_OPS
