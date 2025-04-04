//===-- Compare.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/LLZK/IR/Ops.h"

#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>

#include <concepts>

namespace llzk {

template <typename Op>
concept OpComparable = requires(Op op) {
  { op.getOperation() } -> std::convertible_to<mlir::Operation *>;
};

template <typename Op>
concept NamedOpComparable = requires(Op op) {
  OpComparable<Op>;
  { op.getName() } -> std::convertible_to<mlir::StringRef>;
};

template <OpComparable Op> mlir::FailureOr<bool> isLocationLess(const Op &l, const Op &r) {
  Op &lhs = const_cast<Op &>(l);
  Op &rhs = const_cast<Op &>(r);
  // Try sorting by location first, then name.
  auto lhsLoc = lhs.getOperation()->getLoc().template dyn_cast<mlir::FileLineColLoc>();
  auto rhsLoc = rhs.getOperation()->getLoc().template dyn_cast<mlir::FileLineColLoc>();
  if (lhsLoc && rhsLoc) {
    auto filenameCmp = lhsLoc.getFilename().compare(rhsLoc.getFilename());
    return filenameCmp < 0 || (filenameCmp == 0 && lhsLoc.getLine() < rhsLoc.getLine()) ||
           (filenameCmp == 0 && lhsLoc.getLine() == rhsLoc.getLine() &&
            lhsLoc.getColumn() < rhsLoc.getColumn());
  }
  return mlir::failure();
}

template <OpComparable Op> struct OpLocationLess {
  bool operator()(const Op &l, const Op &r) const { return isLocationLess(l, r).value_or(false); }
};

template <NamedOpComparable Op> struct NamedOpLocationLess {
  bool operator()(const Op &l, const Op &r) const {
    auto res = isLocationLess(l, r);
    if (mlir::succeeded(res)) {
      return res.value();
    }

    Op &lhs = const_cast<Op &>(l);
    Op &rhs = const_cast<Op &>(r);
    return lhs.getName().compare(rhs.getName()) < 0;
  }
};

} // namespace llzk
