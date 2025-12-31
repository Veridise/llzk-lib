//===-- Support.cpp - C API general utilities ---------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/CAPI/Support.h"

/// Note: Duplicated from upstream LLVM. Available in 21.1.8 and later.
void mlirOperationReplaceUsesOfWith(MlirOperation op, MlirValue oldValue, MlirValue newValue) {
  unwrap(op)->replaceUsesOfWith(unwrap(oldValue), unwrap(newValue));
}
