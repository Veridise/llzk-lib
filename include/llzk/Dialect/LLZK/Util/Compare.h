#pragma once

#include "llzk/Dialect/LLZK/IR/Ops.h"

#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>

#include <concepts>

namespace llzk {

template <typename Op>
concept OpComparable = requires(Op op) {
  { op.getOperation() } -> std::convertible_to<mlir::Operation *>;
  { op.getName() } -> std::convertible_to<mlir::StringRef>;
};

template <OpComparable Op> struct OpLocationLess {
  bool operator()(const Op &l, const Op &r) const {
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

    return lhs.getName().compare(rhs.getName()) < 0;
  }
};

} // namespace llzk
