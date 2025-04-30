//===-- OpInterfaces.td ------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_CONSTRAIN_OP_INTERFACES
#define LLZK_CONSTRAIN_OP_INTERFACES

include "mlir/IR/OpBase.td"

def ConstraintOpInterface : OpInterface<"ConstraintOpInterface"> {
  let description = [{
    Common interface for operations that produce a constraint.
  }];
  let cppNamespace = "::llzk::constrain";
}

#endif // LLZK_CONSTRAIN_OP_INTERFACES