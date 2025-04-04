//===-- Attrs.cpp - LLZK Attr method implementations ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/Types.h"

namespace llzk {

mlir::Type FeltConstAttr::getType() const { return FeltType::get(this->getContext()); }

} // namespace llzk
