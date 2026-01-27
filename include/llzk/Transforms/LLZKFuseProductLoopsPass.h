#pragma once

#include <mlir/IR/Region.h>
#include <mlir/Support/LogicalResult.h>

namespace llzk {
/// Identify pairs of scf.for loops that can be fused, fuse them, and then recurse to fuse nested
/// loops
mlir::LogicalResult fuseMatchingLoopPairs(mlir::Region &body, mlir::MLIRContext *context);
} // namespace llzk
