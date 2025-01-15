#pragma once

#include "llzk/Dialect/LLZK/IR/Ops.h"

#include <functional>

namespace llzk {

template <typename Op>
concept OpHashable = requires(Op op) { op.getOperation(); };

template <OpHashable Op> struct OpHash {
  size_t operator()(const Op &op) const {
    return std::hash<mlir::Operation *> {}(const_cast<Op &>(op).getOperation());
  }
};

} // namespace llzk
