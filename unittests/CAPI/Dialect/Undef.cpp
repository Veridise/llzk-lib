//===-- Undef.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Undef.h"

#include "../CAPITestBase.h"

TEST_F(CAPITest, mlir_get_dialect_handle_llzk_undef) {
  {
    mlirGetDialectHandle__llzk__undef__();
  }
}
