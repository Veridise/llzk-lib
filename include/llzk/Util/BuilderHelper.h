//===-- BuilderHelper.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>

namespace llzk {

template <typename OpType, typename... Args>
inline OpType delegate_to_build(mlir::Location location, Args &&...args) {
  mlir::OpBuilder builder(location->getContext());
  return builder.create<OpType>(location, std::forward<Args>(args)...);
}


} // namespace llzk
