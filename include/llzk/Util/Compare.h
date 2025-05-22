//===-- Compare.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

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
  mlir::Location lhsLoc = lhs.getOperation()->getLoc(), rhsLoc = rhs.getOperation()->getLoc();
  auto unknownLoc = mlir::UnknownLoc::get(lhs.getOperation()->getContext());
  // We cannot make judgments on unknown locations.
  if (lhsLoc == unknownLoc || rhsLoc == unknownLoc) {
    return mlir::failure();
  }
  // If we have full locations for both, then we can sort by file name, then line, then column.
  auto lhsFileLoc = llvm::dyn_cast<mlir::FileLineColLoc>(lhsLoc);
  auto rhsFileLoc = llvm::dyn_cast<mlir::FileLineColLoc>(rhsLoc);
  if (lhsFileLoc && rhsFileLoc) {
    auto filenameCmp = lhsFileLoc.getFilename().compare(rhsFileLoc.getFilename());
    return filenameCmp < 0 || (filenameCmp == 0 && lhsFileLoc.getLine() < rhsFileLoc.getLine()) ||
           (filenameCmp == 0 && lhsFileLoc.getLine() == rhsFileLoc.getLine() &&
            lhsFileLoc.getColumn() < rhsFileLoc.getColumn());
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
